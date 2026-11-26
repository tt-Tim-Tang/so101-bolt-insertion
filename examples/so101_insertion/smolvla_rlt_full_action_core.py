"""Native RL-Token Stage-2 heads, replay, and loss with a SmolVLA action map."""

import ast
import enum
import sys
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def load_native(root):
    """Load the exact upstream definitions without importing its Pi0.5 front end."""
    name = "_so101_native_rlt"
    mod = sys.modules[name] = ModuleType(name)
    mod.__dict__.update(torch=torch, nn=nn, Tensor=Tensor, np=np, enum=enum, dataclass=dataclass, field=field,
                        threading=threading, PI05RLTConfig=SimpleNamespace, COLLECTION_PHASE_UNKNOWN=0,
                        COLLECTION_PHASE_WARMUP=1, COLLECTION_PHASE_ONLINE=2)
    selections = {
        "modeling_pi05_rlt.py": ("_build_mlp", "RLTChunkActor", "RLTTwinCritic"),
        "rlt_core.py": ("TransitionSource", "collection_phase_to_id", "_ensure_array", "Transition",
                        "BoundaryContext", "ExecutionWindow", "_resolve_chunk_source"),
        "train_rlt_stage2_pi05.py": ("build_transition", "sample_fixed_std_action", "polyak_update",
                                     "discounted_chunk_return", "ReplayBuffer"),
    }
    for file, wanted in selections.items():
        path = root/file
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = [n for n in tree.body if getattr(n, "name", None) in wanted]
        if {n.name for n in nodes} != set(wanted):
            raise ValueError(f"Missing expected upstream definitions in {file}")
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),mod.__dict__)
    return mod


def transition(native, current, following, actions, sources, *, reward, done, phase, episode, step, success):
    def context(obs, step_id):
        return native.BoundaryContext(z_rl=obs["z_rl"].cpu(), proprio=obs["proprio"].cpu(),
                                      ref_chunk=obs["ref_chunk"].cpu().unsqueeze(0), collection_phase=phase,
                                      episode_id=episode, step_id=step_id)
    window = native.ExecutionWindow(context(current,step))
    for action, source in zip(actions,sources,strict=True):
        window.append(action.cpu(),int(source),source==native.TransitionSource.HUMAN)
    row = native.build_transition(window,context(following,step+len(actions)),horizon=5,action_dim=6,
                                  reward=reward,terminal=done)
    # Preserve the true condition seen at inference. The human replacement is
    # made per step in the imitation target, using the recorded full action.
    row.ref_chunk = current["ref_chunk"].cpu().numpy().copy()
    row.success = int(success)
    row.step_id = step
    return row


class NativeLearner:
    def __init__(self,native,mapping):
        self.native, self.mapping = native,mapping
        self.config = dict(rlt_hidden_dim=512,state_dim=6,action_dim=6,n_action_steps_rl=5,
                           actor_residual_scale=0.,policy_fixed_std=0.,actor_hidden_dim=256,actor_num_layers=2,
                           critic_hidden_dim=256,critic_num_layers=2)
        cfg = SimpleNamespace(**self.config)
        self.actor = native.RLTChunkActor(cfg).cuda()
        assert not self.actor.is_residual
        self.critic = native.RLTTwinCritic(cfg).cuda()
        self.actor_target = deepcopy(self.actor).requires_grad_(False)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.std = .0015/mapping.half
        self.std[-1] = 0.

    def action(self,model,z,proprio,reference,*,noisy):
        mean,_ = model(z,proprio,reference)
        if noisy:
            mean = self.native.sample_fixed_std_action(mean,self.std.expand_as(mean))
        return self.mapping.lock(mean.clamp(-1,1))

    def update(self,replay,step,*,warm):
        native = self.native
        batch = {k:v.cuda() for k,v in replay.sample(256).items()}
        z, proprio, ref = (batch[k] for k in ("z_rl","proprio","ref_chunk"))
        with torch.no_grad():
            next_act = self.action(self.actor_target,batch["next_z_rl"],batch["next_proprio"],
                                   batch["next_ref_chunk"],noisy=True)
            target_q = self.critic_target.q_min(batch["next_z_rl"],batch["next_proprio"],next_act)
            reward, discount = native.discounted_chunk_return(batch["rewards"],batch["valid_mask"],.99)
            target = (reward+discount*(1-batch["done"])*target_q).clamp(-2,2)
        q1,q2 = self.critic(z,proprio,batch["action_chunk"])
        critic_loss = F.mse_loss(q1,target)+F.mse_loss(q2,target)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),10.,error_if_nonfinite=True)
        self.critic_optim.step()
        metrics = dict(update=step+1,critic_loss=critic_loss.item(),replay_q=torch.minimum(q1,q2).mean().item())
        if step%2 == 0:
            drop = torch.rand((len(ref),1,1),device=ref.device)<.5
            conditioned = torch.where(drop,torch.zeros_like(ref),ref)
            mean = self.action(self.actor,z,proprio,conditioned,noisy=False)
            sample = self.mapping.lock(native.sample_fixed_std_action(mean,self.std.expand_as(mean)).clamp(-1,1))
            source = batch["source_chunk"]
            human = (source==native.TransitionSource.HUMAN)|(source==native.TransitionSource.MIXED)
            target_action = torch.where(human[...,None],batch["action_chunk"],ref)
            valid = batch["valid_mask"]
            bc = ((sample-target_action).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1.)
            predicted_deg = torch.rad2deg(self.mapping.physical(mean))
            target_deg = torch.rad2deg(self.mapping.physical(target_action))
            adjacent = valid[:,1:]*valid[:,:-1]
            delta = ((predicted_deg.diff(dim=1)-target_deg.diff(dim=1)).square().mean(-1)*adjacent).sum()
            delta = delta/adjacent.sum().clamp_min(1.)
            self.critic.requires_grad_(False)
            try:
                q = self.critic.q_min(z,proprio,sample).mean()
                wq,wd = (0. if warm else 1.)*q, .02*delta
                total = 10.*bc - wq + wd
                self.actor_optim.zero_grad(set_to_none=True)
                total.backward()
                self.actor_optim.step()
            finally:
                self.critic.requires_grad_(True)
            native.polyak_update(self.actor,self.actor_target,.005)
            native.polyak_update(self.critic,self.critic_target,.005)
            metrics.update(actor_loss=total.item(),bc=bc.item(),actor_q=q.item(),delta_deg2=delta.item(),
                           weighted_delta=wd.item(),weighted_q=wq.item(),reference_dropout=drop.float().mean().item())
            if not warm and abs(wq.item()) > 1e-12:
                metrics["delta_over_abs_q"] = wd.item()/abs(wq.item())
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Nonfinite native RL-Token update {step+1}")
        return metrics

    def checkpoint(self):
        return dict(config=self.config,actor=self.actor.state_dict(),critic=self.critic.state_dict(),
                    actor_target=self.actor_target.state_dict(),critic_target=self.critic_target.state_dict(),
                    actor_optimizer=self.actor_optim.state_dict(),critic_optimizer=self.critic_optim.state_dict())
