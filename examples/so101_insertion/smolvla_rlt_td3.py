"""Single-process adapter for the official RLinf RLT-TD3 model and loss mixins."""

import ast
import json
import random
import sys
from copy import deepcopy
from importlib import import_module
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def load_rlinf(root):
    # The head has no distributed dependencies. Import its real dependencies
    # without initializing the Ray scheduler at the parent package level.
    for name in ("rlinf", "rlinf.models", "rlinf.models.embodiment",
                 "rlinf.models.embodiment.mlp_policy", "rlinf.models.embodiment.modules"):
        module = ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        sys.modules[name] = module
    head = import_module("rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy")
    scope = dict(torch=torch, F=F, ForwardType=head.ForwardType,
                 Worker=SimpleNamespace(timer=lambda _: lambda fn: fn),
                 use_simulator_transition_replay=lambda _cfg: True)
    # Load the upstream loss classes verbatim. The source files also contain
    # FSDP/Ray workers, which the single-process smoke run does not instantiate.
    worker_dir = root / "rlinf/workers/actor"
    for file, name in (("fsdp_rlt_ac_policy_worker.py", "RLTACLossMixin"),
                       ("fsdp_rlt_td3_policy_worker.py", "RLTTD3LossMixin")):
        path = worker_dir / file
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        exec(compile(ast.Module(body=[mixin], type_ignores=[]), str(path), "exec"), scope)
    return head.RLTTD3MLPPolicy, scope["RLTTD3LossMixin"]


def batch_replay(replay, device):
    return {
        **{
            field: {key: torch.stack([row[field][key] for row in replay])
                    for key in ("z_rl", "proprio", "ref_chunk")}
            for field in ("curr_obs", "next_obs")
        },
        **{key: torch.as_tensor(np.array([row[key] for row in replay]), device=device)
           for key in ("rewards", "dones", "terminations", "intervene_flags")},
        "actions": torch.stack([row["actions"] for row in replay]),
    }


def actor_delta_rad(policy, obs, actions):
    predicted, _, _ = policy.sac_forward(obs, deterministic=True, apply_action_noise=False)
    return (predicted.reshape_as(obs["ref_chunk"]) - obs["ref_chunk"]).abs()[..., :5] * actions.half[:5]


def learner(model_class, loss_mixin, actions, args, alg, actor, device):
    chunk, action_dim = args["num_action_chunks"], args["action_dim"]
    policy = model_class(**args).to(device)
    target = deepcopy(policy).requires_grad_(False)

    def q_gripper_lock(_module, inputs):
        state, commands = inputs
        return state, actions.lock(commands.reshape(-1, chunk, action_dim)).flatten(1)

    if not args.get("critic_in_residual_space", False):
        for network in (policy, target):
            network.q_head.register_forward_pre_hook(q_gripper_lock)

    class Losses(loss_mixin):
        torch_dtype = torch.float32

        def _env_values(self, name, _index, _dtype):
            return self.cfg.algorithm[name]

    loss = Losses()
    loss.cfg = SimpleNamespace(
        actor=SimpleNamespace(model=SimpleNamespace(num_action_chunks=chunk, action_dim=action_dim)),
        algorithm={**alg, "q_head_type": actor["model"]["q_head_type"]},
    )
    loss.model, loss.target_model = policy, target
    loss.qf_optimizer = torch.optim.Adam(policy.q_head.parameters(), lr=actor["critic_optim"]["lr"])
    return loss, torch.optim.Adam(policy.actor.parameters(), lr=actor["optim"]["lr"])


def take_batch(value, indices):
    if isinstance(value, dict):
        return {key: take_batch(tensor, indices) for key, tensor in value.items()}
    return value[indices]


def train_step(loss, actor_optim, actor, batch, step, *, update_actor=True):
    policy, target, alg = loss.model, loss.target_model, loss.cfg.algorithm
    q_optim = loss.qf_optimizer
    loss.update_step = step
    q_optim.zero_grad(set_to_none=True)
    q_loss, q_stats = loss.forward_critic(batch)
    q_loss.backward()
    q_grad = torch.nn.utils.clip_grad_norm_(policy.q_head.parameters(), actor["critic_optim"]["clip_grad"],
                                           error_if_nonfinite=True)
    q_optim.step()
    row = dict(step=step + 1, critic_loss=q_loss.item(), q_grad=q_grad.item(), **q_stats)
    if update_actor and step % alg["critic_actor_ratio"] == 0:
        actor_optim.zero_grad(set_to_none=True)
        objective, _, stats = loss.forward_actor(batch)
        objective.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), actor["optim"]["clip_grad"],
                                              error_if_nonfinite=True)
        actor_optim.step()
        row.update(actor_loss=objective.item(), actor_grad=grad.item(), **stats)
    if step % alg["target_update_freq"] == 0:
        with torch.no_grad():
            for dst, src in zip(target.parameters(), policy.parameters(), strict=True):
                dst.lerp_(src, alg["tau"])
    if not all(np.isfinite(value) for value in row.values()):
        raise RuntimeError(f"Nonfinite TD3 metric at update {step + 1}: {row}")
    return row


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panels = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for axis, keys in zip(panels, (("critic_loss",), ("actor_loss", "bc_loss"),
                                  ("q_data", "q_pi", "q_value_0", "q_value_1")), strict=True):
        for key in keys:
            points = [(row["step"], row[key]) for row in rows if key in row]
            axis.plot(*zip(*points), label=key)
        axis.legend()
        axis.grid(alpha=.25)
    panels[-1].set_xlabel("Critic update")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def update_td3(rlinf, actions, replay, output, *, updates=50, device="cuda", offline_run=None):
    model_class, loss_mixin = load_rlinf(rlinf.resolve())
    template = yaml.safe_load(
        (rlinf / "examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml").read_text(encoding="utf-8")
    )
    actor = template["actor"]
    alg = template["algorithm"]
    args = {key: actor["model"][key] for key in ("actor_noise_sigma", "mlp_hidden_dim", "mlp_num_hidden_layers")}
    if offline_run is not None:
        args["actor_noise_sigma"] = actions.sigma
        schedule = alg["actor_weight_schedule"]
        for key in ("warmup_updates", "ramp_updates"):
            schedule[key] //= 10
    sample = replay[0]["curr_obs"]
    chunk, action_dim = sample["ref_chunk"].shape
    args.update(z_dim=sample["z_rl"].numel(), proprio_dim=sample["proprio"].numel(),
                action_dim=action_dim, num_action_chunks=chunk, ref_num_action_chunks=chunk)
    torch.manual_seed(actor["seed"])
    rng = random.Random(actor["seed"])
    loss, actor_optim = learner(model_class, loss_mixin, actions, args, alg, actor, device)
    policy, target, q_optim = loss.model, loss.target_model, loss.qf_optimizer

    warm = batch_replay(replay, device)
    if offline_run is not None:
        saved_replay = {
            key: {name: tensor.cpu() for name, tensor in value.items()} if isinstance(value, dict) else value.cpu()
            for key, value in warm.items()
        }
        path = output / "replay.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(saved_replay, temporary)
        temporary.replace(path)
    with torch.no_grad():
        obs = {key: value[:actor["micro_batch_size"]] for key, value in warm["curr_obs"].items()}
        first, _, _ = policy.sac_forward(obs, deterministic=True, apply_action_noise=False)
        first = first.reshape(-1, chunk, action_dim)
        delta = (first - obs["ref_chunk"]).abs()
        initial = dict(actor_ref_mean_abs_normalized=delta.mean().item(),
                       arm_mean_abs_rad=(delta[..., :5] * actions.half[:5]).mean().item(),
                       arm_max_abs_rad=(delta[..., :5] * actions.half[:5]).max().item())
    (output / "actor-initial.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps(initial), flush=True)

    rows = []
    with (output / "td3-curve.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for step in range(updates):
            indices = rng.choices(range(len(replay)), k=actor["global_batch_size"])
            row = train_step(loss, actor_optim, actor, take_batch(warm, indices), step)
            rows.append(row)
            print(json.dumps(row), file=log)
            if step == 0 or (step + 1) % (100 if offline_run is not None else 10) == 0:
                print(json.dumps(row), flush=True)

    if offline_run is not None:
        path = output / "td3-offline.pt"
        payload = dict(
            model=policy.state_dict(), target=target.state_dict(), model_config=args, algorithm=loss.cfg.algorithm,
            critic_optimizer=q_optim.state_dict(), actor_optimizer=actor_optim.state_dict(),
            update_step=updates, run=offline_run, replay_path="replay.pt",
            replay_rng=rng.getstate(), torch_rng=torch.random.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
        )
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        reloaded = model_class(**saved["model_config"]).to(device).eval().requires_grad_(False)
        reloaded.load_state_dict(saved["model"], strict=True)
        target.load_state_dict(saved["target"], strict=True)
        q_optim.load_state_dict(saved["critic_optimizer"])
        actor_optim.load_state_dict(saved["actor_optimizer"])
        restored_replay = torch.load(output / saved["replay_path"], map_location=device, weights_only=True)
        for key in ("actions", "rewards", "dones"):
            torch.testing.assert_close(restored_replay[key], warm[key], atol=0, rtol=0)
        with torch.no_grad():
            torch.testing.assert_close(actor_delta_rad(reloaded, obs, actions),
                                       actor_delta_rad(policy, obs, actions), atol=0, rtol=0)
            delta = actor_delta_rad(reloaded, restored_replay["curr_obs"], actions)
        tail = rows[-100:]
        avg = {key: float(np.mean([row[key] for row in tail if key in row]))
               for key in ("critic_loss", "actor_loss", "bc_loss", "q_data", "q_pi")}
        final_actor = next(row for row in reversed(rows) if "actor_loss" in row)
        summary = dict(
            critic_updates=updates, actor_updates=sum("actor_loss" in row for row in rows),
            critic_loss=rows[-1]["critic_loss"], actor_loss=final_actor["actor_loss"],
            actor_loss_step=final_actor["step"], last_100_critic_steps_mean=avg,
            warmup_arm_mean_abs_rad=delta.mean().item(), warmup_arm_max_abs_rad=delta.max().item(),
            reload_verified=True,
        )
        (output / "offline.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)
        plot(rows, output / "td3-curve.png")
        return reloaded

    torch.save(dict(model=policy.state_dict(), target=target.state_dict(), model_config=args),
               output / "td3-smoke.pt")
    plot(rows, output / "td3-curve.png")
