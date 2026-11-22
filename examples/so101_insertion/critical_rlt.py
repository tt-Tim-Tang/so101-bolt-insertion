"""Small paired RLT experiment starting at held-bolt states, without interventions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTChunkActor, RLTokenEncoder, RLTTwinCritic

from .critical_states import restore
from .online import CHUNK, Policy, diffusion_noise
from .scene import Insertion


GAMMA, TAU, BATCH = .99, .005, 256


class Learner:
    def __init__(self, config):
        self.actor = RLTChunkActor(config).cuda()
        self.critic = RLTTwinCritic(config).cuda()
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.replay = None
        self.updates = 0

    def append(self, rows):
        data = [torch.stack(column).cuda() for column in zip(*rows, strict=True)]
        self.replay = data if self.replay is None else [torch.cat([x,y]) for x,y in zip(self.replay,data,strict=True)]

    def update(self, count):
        for _ in range(count):
            idx = torch.randint(len(self.replay[0]), (BATCH,), device="cuda")
            z, s, ref, act, zn, sn, rn, reward, discount, valid = (v[idx] for v in self.replay)
            with torch.no_grad():
                m, std = self.target_actor(zn, sn, rn)
                target = reward + discount * self.target_critic.q_min(zn, sn, m + std*torch.randn_like(m))
            q1, q2 = self.critic(z, s, act)
            self.critic_opt.zero_grad(set_to_none=True)
            (F.mse_loss(q1, target) + F.mse_loss(q2, target)).backward()
            self.critic_opt.step()
            if self.updates % 2 == 0:
                masked = ref * (torch.rand(len(ref), 1, 1, device="cuda") >= .5)
                mean, std = self.actor(z, s, masked)
                proposed = mean + std*torch.randn_like(mean)
                bc = ((proposed-ref).square().mean(-1) * valid).sum()/valid.sum()
                self.critic.requires_grad_(False)
                loss = bc - self.critic.q_min(z, s, proposed).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.actor_opt.step()
                self.critic.requires_grad_(True)
            with torch.no_grad():
                for src, dst in ((self.actor,self.target_actor),(self.critic,self.target_critic)):
                    torch._foreach_lerp_(list(dst.parameters()), list(src.parameters()), TAU)
            self.updates += 1


def replay_rows(points, actions, success):
    # Actions and rewards are indexed by physical control step. Every even start
    # has its own observed state and fresh VLA reference, including overlapping windows.
    n = len(actions)
    rows = []
    for t, start in points.items():
        end = min(t+CHUNK, n)
        k = end-t
        if k <= 0:
            continue
        terminal = end == n
        following = start if terminal else points[end]
        block = torch.stack(actions[t:end] + [actions[end-1]]*(CHUNK-k))
        reward = float(terminal and success)*GAMMA**(k-1)
        discount = 0. if terminal else GAMMA**k
        rows.append((*start, block, *following, torch.tensor([reward]), torch.tensor([discount]),
                     (torch.arange(CHUNK)<k).float()))
    return rows


@torch.no_grad()
def episode(env, policy, actor, seed, start, collect, explore, trace=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    aux = torch.Generator(device="cuda").manual_seed(seed+1000000)
    if start is None:
        env.reset(seed)
    else:
        restore(env, start)
        gen.set_state(torch.from_numpy(start["policy_rng"]))
    def observe(generator, token):
        cfg = policy.policy.config
        noise = diffusion_noise(cfg, generator)
        return policy.decide(env.observation(), token, noise=noise)

    points, actions = {}, []
    lo = policy.joint_targets(-torch.ones(1,6))[0]
    hi = policy.joint_targets(torch.ones(1,6))[0]
    if not np.all(hi > lo):
        raise ValueError("Cannot invert action normalization")
    active = start is not None
    actor_chunks = 0
    max_error = 0.
    first_historical_difference = None
    while True:
        active |= env.info()["critical"] and env.pickup_state()[-1]
        if trace is not None:
            observation = env.observation()
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.set_rng_state(gen.get_state())
                original = policy.decide(observation, False)
            point = policy.decide(observation, False, noise=diffusion_noise(policy.policy.config, gen))
            error = (original.reference-point.reference).abs().max().item()
            if error > 1e-6:
                raise AssertionError(f"Same-frame policy output differs at step {env.steps}: {error}")
            max_error = max(max_error, error)
        else:
            point = observe(gen, collect or (active and actor is not None))
        command = point.reference
        if active and actor is not None:
            m, std = actor(*(x[None].cuda() for x in point))
            command = (m + std*torch.randn_like(m) if explore else m)[0].cpu()
            actor_chunks += 1
        if collect:
            points[env.steps] = point
        targets = policy.joint_targets(command)
        if trace is not None and env.steps//CHUNK < len(trace):
            expected = trace[env.steps//CHUNK]["targets"].numpy()
            if first_historical_difference is None and np.max(np.abs(targets-expected)) > 1e-6:
                first_historical_difference = env.steps
        for ix, target in enumerate(targets):
            if collect and ix and ix % 2 == 0:
                points[env.steps] = observe(aux, True)
            _, win, timeout, _ = env.step(target)
            if collect:
                normalized = command[ix].clone()
                clipped = env.data.ctrl != target
                if clipped.any():
                    normalized[clipped] = torch.from_numpy(((env.data.ctrl[clipped]-lo[clipped]) /
                                                            (hi[clipped]-lo[clipped])*2-1).astype(np.float32))
                actions.append(normalized)
            if win or timeout:
                result = dict(seed=seed, steps=env.steps, success=win, actor_chunks=actor_chunks)
                if trace is not None:
                    result.update(max_same_frame_action_error=max_error,
                                  first_historical_difference=first_historical_difference)
                return result, (
                    replay_rows(points, actions, win) if collect else [])


def run(args):
    torch.manual_seed(42)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.json").write_text(json.dumps(dict(
        train_starts=24, test_starts=30, passes=["SFT replay prefill", "online actor"],
        max_critical_steps=250, control_hz=50, chunk=CHUNK, replay_stride=2, utd=5,
        batch=BATCH, gamma=GAMMA, tau=TAU, critic_per_actor=2, lr=3e-4, bc_weight=1,
        q_weight=1, reference_dropout=.5, seed=42, expert=False, **vars(args)
    ), default=str, indent=2))
    src = torch.load(args.stage1, map_location="cpu", weights_only=False)
    assert src["image_only"]
    cfg = PI05RLTConfig(**{**src["config"], "device":"cuda", "n_action_steps_rl":CHUNK})
    encoder = RLTokenEncoder(cfg)
    encoder.load_state_dict(src["encoder_state_dict"])
    encoder.cuda().eval().requires_grad_(False)
    policy = Policy(args.pi05, encoder, cfg.rlt_hidden_dim, "cuda", args.tokenizer, CHUNK)
    learner = Learner(cfg)
    train, test = [[dict(np.load(path)) for path in sorted((args.bank/group).glob("*.npz"))]
                   for group in ("train", "test")]
    manifest = json.loads((args.bank/"manifest.json").read_text())
    for group, states, seeds in (("train", train, range(3000,3024)), ("test", test, range(6000,6030))):
        expected = manifest[group]
        assert set(map(int,expected)) == set(seeds)
        assert {int(x["seed"]) for x in states} == {int(k) for k,v in expected.items() if v is not None}
    env = Insertion(max_steps=250, control_fps=50)
    def run_set(name, states, actor=None, *, collecting=False):
        results = []
        with (out/f"{name}.jsonl").open("x") as log:
            for s in states:
                seed = int(s["seed"]) if isinstance(s, dict) else s
                metrics, rows = episode(env, policy, actor, seed, s if isinstance(s, dict) else None,
                                        collecting, collecting and actor is not None)
                if collecting:
                    learner.append(rows)
                    if actor is not None:
                        learner.update(5*len(rows))
                results.append(metrics)
                print(json.dumps(dict(**metrics, transitions=len(rows), updates=learner.updates)), file=log, flush=True)
                print(name,seed,metrics["steps"],int(metrics["success"]),flush=True)
        return results
    def paired(base, actor):
        assert [x["seed"] for x in base] == [x["seed"] for x in actor]
        return dict(n=len(base), sft=sum(x["success"] for x in base), actor=sum(x["success"] for x in actor),
                    gains=[x["seed"] for x,y in zip(actor,base) if x["success"] and not y["success"]],
                    losses=[x["seed"] for x,y in zip(actor,base) if y["success"] and not x["success"]])
    try:
        saved = torch.load(args.bank/"reference.pt", map_location="cpu", weights_only=True)
        env.max_steps = 1000
        check, _ = episode(env, policy, None, 10000, None, False, False, saved["trace"])
        (out/"alignment.json").write_text(json.dumps(check,indent=2))
        env.max_steps = 250
        run_set("warmup", train, collecting=True)
        learner.update(5*len(learner.replay[0]))
        run_set("online", train, learner.actor, collecting=True)
        learner.actor.eval().requires_grad_(False)
        torch.save(dict(actor=learner.actor.state_dict(), updates=learner.updates), out/"actor.pt")
        result = {"critical":paired(run_set("sft", test), run_set("actor", test, learner.actor))}
        (out/"result.json").write_text(json.dumps(result,indent=2))
        if result["critical"]["actor"] > result["critical"]["sft"]:
            env.max_steps = 1000
            seeds = range(10000,10030)
            result["full"] = paired(run_set("full-sft",seeds),run_set("full-actor",seeds,learner.actor))
            (out/"result.json").write_text(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pi05","stage1","bank","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    run(parser.parse_args())
