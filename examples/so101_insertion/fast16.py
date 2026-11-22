"""Small new-camera verification with a frozen 16/10 split and no seating corrections."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from .wide_env import WideInsertion, expert


def make_cases():
    rng = np.random.default_rng(916182)
    def case(target):
        return dict(source=[0, 0], target=list(target), joints=[0]*5, noise=int(rng.integers(2**31)))
    return dict(train=[case(xy) for xy in product((-13.5, -4.5, 4.5, 13.5), repeat=2)],
                validation=[case(xy) for xy in rng.uniform(-18, 18, size=(10, 2))])


def preflight(root):
    root.mkdir(parents=True, exist_ok=False)
    cases = make_cases()
    (root/"splits.json").write_text(json.dumps(cases, indent=2))
    env = WideInsertion(target_bore_radius=.0057)
    try:
        with (root/"expert-check.jsonl").open("x", buffering=1) as log:
            for group, group_cases in cases.items():
                for i, case in enumerate(group_cases):
                    try:
                        result = expert(env, case, seat_correction=False)
                    except RuntimeError as exc:
                        result = dict(success=False, pickup=env.pick, arrival=env.arrive, error=str(exc))
                    print(json.dumps(dict(group=group,index=i,steps=env.steps,case=case,**result)),file=log)
                    print(group, i, env.steps, result.get("pickup"), result.get("arrival"), result["success"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    preflight(args.out)
