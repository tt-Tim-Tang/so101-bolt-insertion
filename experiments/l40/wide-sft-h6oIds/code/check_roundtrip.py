"""Verify that saved images, joint states and action timing reproduce the training simulator."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from examples.so101_insertion.wide_env import WideInsertion

root=Path(sys.argv[1])
attempts=[json.loads(s) for s in (root/"collection.jsonl").read_text().splitlines()]
cases=[a["case"] for a in attempts if a["result"]["success"]]
ids=[0,7,27,56,63]
files=sorted((root/"dataset/data").rglob("*.parquet"))
actions=pq.read_table(files,columns=["episode_index","frame_index","action"],
                      filters=[("episode_index","in",ids)]).sort_by([("episode_index","ascending"),("frame_index","ascending")])
frames=pq.read_table(files,filters=[("episode_index","in",ids),("frame_index","in",[0,94,129,193,325,395])])
frames={(r["episode_index"],r["frame_index"]):r for r in frames.to_pylist()}
rows=actions.to_pylist()
env=WideInsertion()
summary=[]
try:
    for ep in ids:
        env.start(cases[ep])
        for row in (r for r in rows if r["episode_index"]==ep):
            tick=row["frame_index"]
            stored=frames.get((ep,tick))
            if stored:
                live=env.observation()
                delta={"episode":ep,"frame":tick,"state_max_rad":float(np.max(np.abs(
                    np.asarray(stored["observation.state"])-live["observation.state"])))}
                for camera in ("top","wrist"):
                    key=f"observation.images.{camera}"
                    rgb=np.asarray(Image.open(io.BytesIO(stored[key]["bytes"])).convert("RGB"))
                    diff=np.abs(rgb.astype(float)-live[key])
                    delta[camera+"_mean_abs_pixel"]=float(diff.mean())
                    delta[camera+"_max_abs_pixel"]=float(diff.max())
                summary.append(delta)
            _,_,_,info=env.step(row["action"])
        assert info["success"]
finally:
    env.close()
(root/"roundtrip.json").write_text(json.dumps(summary,indent=2))
