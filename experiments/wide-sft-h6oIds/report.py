"""Summarize only complete, independently logged validation and test episodes."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics(rows):
    return {key:sum(bool(row[key]) for row in rows) for key in ("pickup","arrival","alignment","success")}


def failures(rows):
    return dict(Counter(next((key for key in ("pickup","arrival","alignment") if not r[key]),"insertion")
                        for r in rows if not r["success"]))


p=argparse.ArgumentParser()
p.add_argument("root",type=Path)
root=p.parse_args().root
splits=json.loads((root/"splits.json").read_text())
attempts=read_lines(root/"collection.jsonl")
training=[r for r in attempts if r["result"]["success"]]
assert [r["cell"] for r in training]==list(range(64))
for key in ("source_cell","target_cell"):
    assert len({tuple(r["case"][key]) for r in training})==64
def identity(case):
    return tuple(case["source"]+case["target"]+case["joints"])
identities=[identity(r["case"]) for r in training]+[identity(c) for s in ("validation","test") for c in splits[s]]
assert len(set(identities))==144
metadata=json.loads((root/"dataset/meta/info.json").read_text())
assert metadata["total_episodes"]==64 and metadata["fps"]==50
doc=dict(frames=metadata["total_frames"],attempts=len(attempts),training_episodes=len(training),
         demo_failures=dict(Counter(r["result"].get("reason",r["result"]["phase"])
                                    for r in attempts if not r["result"]["success"])))
for stage,n in ((2000,30),(4000,30),(6000,30),("final-test",50)):
    name=f"validation-{stage}" if isinstance(stage,int) else stage
    rows=read_lines(root/(name+".jsonl"))
    assert sorted(r["index"] for r in rows)==list(range(n))
    assert all(r["case"]==splits["validation" if n==30 else "test"][r["index"]] for r in rows)
    doc[name]=dict(n=n,**metrics(rows),failures=failures(rows))
fixed=read_lines(root/"fixed-validation.jsonl")
doc["fixed"]={}
for mode in ("full","after_pick"):
    rows=[r for r in fixed if r["mode"]==mode]
    assert sorted(r["index"] for r in rows)==list(range(30))
    doc["fixed"][mode]=dict(n=len(rows),setup_failures=sum(r["setup_failed"] for r in rows),
                            successes=sum(r["success"] for r in rows))
probe=read_lines(root/"image-intervention.jsonl")
values=[r["signed_mm"] for r in probe if r["available"]]
doc["visual_probe"]=dict(total=len(probe),available=len(values),positive=sum(v>0 for v in values),
                         median_signed_mm=median(values) if values else None)
doc["selection"]=json.loads((root/"selection.json").read_text())
(root/"results.json").write_text(json.dumps(doc,indent=2))
