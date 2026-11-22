from pathlib import Path
from statistics import median
import json

root=Path(__file__).resolve().parent
steps=(2000,4000,6000)
def read(name):
    return [json.loads(r) for r in (root/name).read_text().splitlines()]
rows={s:read(f"chunks-{s}.jsonl") for s in steps}
assert all(len(v)==96 and len({(x["e"],x["t"]) for x in v})==96 for v in rows.values())
ticks=(35,75,94,110,118,125)
metrics={}
for s,a in rows.items():
    metrics[s]={}
    for t in ticks:
        b=[x for x in a if x["t"]==t]
        metrics[s][t]=dict(xy_mm=median(x["xy_mm"] for x in b),abs_z_mm=median(abs(x["z_err_mm"]) for x in b),
                           abs_measured_gripper_error_rad=median(abs(x["grip_rad"]) for x in b))
    for key in ("cmd","measured"):
        b=[x for x in a if x["t"] in (110,118) and x["ref_"+key+"_cross"] is not None and x["ref_"+key+"_cross"]>0]
        paired=[x for x in b if x["pred_"+key+"_cross"] is not None]
        delta=[x["pred_"+key+"_cross"]-x["ref_"+key+"_cross"] for x in paired]
        metrics[s][key+"_cross"]=dict(expert_windows=len(b),cross_in_window=len(paired),
                                       median_delta_ticks=median(delta) if delta else None)
take=read("takeover.jsonl")
assert len(take)==32
results={}
for mode in ("expert","model","expert_arm","expert_grip"):
    b=[x for x in take if x["mode"]==mode]
    results[mode]=dict(pick=sum(x["pick"] for x in b),n=len(b),close_ticks=[x["close"] for x in b],
                       measured_ticks=[x["measured"] for x in b])
path=json.loads((root/"path.json").read_text())
summary=dict(steps=metrics,takeover=results,interface_checked=len(path),
             max_interface_action_diff_rad=max(x["action_max_rad"] for x in path))
(root/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
