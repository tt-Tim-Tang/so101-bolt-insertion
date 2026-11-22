"""Run the interactive insertion trainer with a geometry-based simulated operator."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def decide(state, previous):
    """Choose a block-boundary command from this episode's observed states and commands."""
    def command(action, blocks, reason):
        return dict(request=state["request"], action=action, blocks=blocks, reason=reason)

    step = state["step"]
    radius, depth, tilt = (state[k] for k in ("radial_error_m", "insertion_depth_m", "tilt_rad"))
    entered = state["critical"] or any(e["state"]["critical"] for e in previous)
    last = previous[-1]["state"] if previous else None
    if not state["bolt_at_jaws"] or not entered:
        if last and not last["bolt_at_jaws"] and not state["bolt_at_jaws"]:
            return command("finish", 0, "Two observations without a held bolt; the insertion operator cannot regrasp.")
        if not entered and step >= 310:
            return command("finish", 0, "The bolt has not reached the insertion region after the pickup check.")
        return command("policy", 2, "Recheck the grasp and approach before declaring a failed pickup.")
    if not last:
        return command("policy", 3 if depth < -.012 else 1, "Allow the held bolt to approach under policy control.")

    takeovers = sum(e["command"]["action"] == "align" and not e["state"]["human_active"] for e in previous)
    if state["human_active"]:
        if takeovers >= 2 and tilt >= .48:
            return command("finish", 0, "Tilt persists after two takeovers; translation cannot fix the slipping grasp.")
        if radius <= .002:
            return command("policy", 1, "Bolt tip centered; hand back for a short autonomous attempt.")
        if not last["human_active"] and last["radial_error_m"] - radius >= .001:
            return command("align", 2, "Lateral alignment improved; refine once using the updated grasp offset.")
        if takeovers >= 2:
            return command("finish", 0, "Second alignment could not restore a centered approach.")
        return command("policy", 1, "Alignment stopped improving; return control and reassess.")

    # A 1 mm radius change or 2 mm depth gain is observable progress at this scale.
    recent = [e["state"] for e in previous if step - e["state"]["step"] <= 50]
    stalled = bool(recent and step - recent[0]["step"] >= 40
                   and max(s["radial_error_m"] for s in recent) - radius < .001
                   and depth - min(s["insertion_depth_m"] for s in recent) < .002)
    near_mouth = depth > -.012
    drifting = radius - last["radial_error_m"] >= .003
    if stalled or (near_mouth and radius >= .008 and (drifting or tilt >= .32)):
        if takeovers >= 2:
            return command("finish", 0, "The policy diverged or stalled again after both alignment attempts.")
        return command("align", 3, "Held bolt is drifting at the mouth or has stopped progressing; lift and recenter.")
    if depth >= .016 or (near_mouth and radius > .004):
        return command("policy", 1, "Check again after ten steps near the rim or insertion depth.")
    return command("policy", 2, "The approach still has room or is progressing; keep policy control.")


def serve(output, alive):
    """Attach to one supervised training run; exit when its trainer stops or summary exists."""
    output = Path(output)
    state_path = output / "operator-state.json"
    events_path = output / "operator-events.jsonl"
    command_path = output / "operator-command.json"
    history = []
    if events_path.exists():
        history = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    last_request = history[-1]["state"]["request"] if history else None
    checked = False
    while alive() and not (output / "summary.json").exists():
        if state_path.exists() and not command_path.exists():
            if not checked:
                config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
                if config["inference_only"] or not config["resume_from"]:
                    raise ValueError("The automatic operator requires a resumed training run.")
                checked = True
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["request"] != last_request:
                history = [e for e in history if e["state"]["seed"] == state["seed"]]
                cmd = decide(state, history)
                cmd["operator"] = "scripted_geometry"
                temp = output / "automatic-command.tmp"
                temp.write_text(json.dumps(cmd), encoding="utf-8")
                temp.replace(command_path)
                history.append(dict(state=state, command=cmd))
                last_request = state["request"]
        time.sleep(.15)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operator-resume", required=True)
    parser.add_argument("--evaluate")
    args, _ = parser.parse_known_args()
    if args.evaluate or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Use a new training output directory for the automatic operator.")
    with subprocess.Popen([sys.executable, "-m", "examples.so101_insertion.online", *sys.argv[1:]]) as trainer:
        try:
            serve(args.output, lambda: trainer.poll() is None)
            return trainer.wait()
        finally:
            if trainer.poll() is None:
                trainer.terminate()


if __name__ == "__main__":
    sys.exit(main())
