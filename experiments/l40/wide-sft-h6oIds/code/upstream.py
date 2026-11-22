"""Run the upstream π0.5 trainer or embedding exporter with checked weight loading."""

import os
import runpy
import sys
from pathlib import Path

from .scene import ROOT


def main():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if len(sys.argv) < 2 or sys.argv[1] not in {"finetune", "embeddings"}:
        raise SystemExit("Usage: python -m examples.so101_insertion.upstream {finetune,embeddings} [args]")
    stage = sys.argv.pop(1)
    original_load = PI05Policy.load_state_dict
    original_from_pretrained = PI05Policy.from_pretrained.__func__

    def checked_load(policy, state_dict, strict=True, **kwargs):
        result = original_load(policy, state_dict, strict=True, **kwargs)
        policy._insertion_weights_loaded = True
        return result

    def checked_from_pretrained(cls, *args, **kwargs):
        policy = original_from_pretrained(cls, *args, **kwargs)
        if not getattr(policy, "_insertion_weights_loaded", False):
            raise RuntimeError("π0.5 pretrained weights did not load. See the loader error above.")
        return policy

    PI05Policy.load_state_dict = checked_load
    PI05Policy.from_pretrained = classmethod(checked_from_pretrained)
    if stage == "finetune":
        from lerobot.scripts import lerobot_train

        if optimizer_dir := os.environ.get("SO101_OPTIMIZER_STATE_DIR"):
            from .checkpoint_storage import configure_checkpoint_storage

            configure_checkpoint_storage(lerobot_train, Path(optimizer_dir))
        lerobot_train.main()
    else:
        script = Path(ROOT, "tmp/RL-Token-Pi05-open/scripts/precompute_pi05_embeddings.py")
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
