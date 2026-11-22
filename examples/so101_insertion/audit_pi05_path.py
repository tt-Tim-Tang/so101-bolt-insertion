"""Compare Stage 2 targets to the standalone SFT path on the exact frames it receives."""
import argparse
import json
from pathlib import Path
import sys
from threading import local
from unittest.mock import patch

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors

from . import online
from .scene import JOINTS, TASK
from .stage2_sim import Simulator


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    saved = torch.load(args.trace, map_location="cpu", weights_only=True)
    out = Path(remaining[remaining.index("--output")+1])
    tokenizer = remaining[remaining.index("--tokenizer")+1]
    frames = local()
    chunks = []
    source_input = Simulator.sft_input
    load_policy = Simulator.load_policy
    send = Simulator.send_action

    def keep_frame(obs, state):
        frames.raw = obs
        frames.state = state
        return source_input(obs, state)

    def load(checkpoint, device):
        model = load_policy(checkpoint, device)
        pre, post = make_pre_post_processors(model.config, str(checkpoint), preprocessor_overrides={
            "device_processor":{"device":str(device)}, "tokenizer_processor":{"tokenizer_name":tokenizer}})
        predict = model.predict_action_chunk

        def compare(batch, **kwargs):
            i = len(chunks)
            noise = kwargs["noise"]
            if i < len(saved["trace"]):
                assert torch.equal(noise.cpu(), saved["trace"][i]["noise"])
            frame = {"task": TASK, "observation.state": frames.state.detach().cpu()}
            for k in online.CAMERA_MAP:
                image = torch.from_numpy(frames.raw[k.rsplit(".",1)[-1]])
                frame[k] = image.permute(2,0,1).float()/255
            original = pre(frame)
            errors = {k: float((v.float()-batch[k].float()).abs().max()) for k,v in original.items()
                      if torch.is_tensor(v)}
            ref = predict(original, noise=noise)
            actual = predict(batch, **kwargs)
            row = dict(expected=post(ref[:,:online.CHUNK].float().cpu())[0].cpu().numpy(),
                       raw_error=float((ref-actual).abs().max()), target_error=0., historical_error=None,
                       input_errors=errors)
            with (out/"input-audit.jsonl").open("a") as log:
                print(json.dumps(dict(chunk=i, errors=errors, raw_error=row["raw_error"])), file=log)
            chunks.append(row)
            return actual
        model.predict_action_chunk = compare
        return model

    def check(sim, target):
        i, offset = divmod(sim.env.steps, online.CHUNK)
        values = np.array([target[f"{j}.pos"] for j in JOINTS])
        row = chunks[i]
        err = float(np.max(np.abs(values-row["expected"][offset])))
        row["target_error"] = max(row["target_error"],err)
        if i < len(saved["trace"]):
            old = saved["trace"][i]["targets"][offset].numpy()
            row["historical_error"] = max(row["historical_error"] or 0.,float(np.max(np.abs(values-old))))
        return send(sim,target)

    with patch.object(sys,"argv",[sys.argv[0],*remaining]), patch.object(Simulator,"sft_input",staticmethod(keep_frame)), \
            patch.object(Simulator,"load_policy",staticmethod(load)), patch.object(Simulator,"send_action",check):
        online.main()
    episode = json.loads((out/"episodes.jsonl").read_text())
    result = dict(seed=episode["seed"],steps=episode["steps"],success=episode["success"],chunks=len(chunks),
                  max_same_frame_raw_error=max(x["raw_error"] for x in chunks),
                  max_same_frame_target_error_rad=max(x["target_error"] for x in chunks),
                  historical_steps=saved["steps"], historical_success=saved["success"],
                  first_historical_difference=next((i*online.CHUNK for i,x in enumerate(chunks)
                                                     if (x["historical_error"] or 0.)>1e-6),None))
    (out/"alignment.json").write_text(json.dumps(result,indent=2))
    if result["max_same_frame_target_error_rad"] > 1e-6:
        raise AssertionError(result)


if __name__ == "__main__":
    main()
