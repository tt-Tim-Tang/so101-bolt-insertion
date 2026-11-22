"""Extract paired visual observations from the five complete review videos."""
import argparse
from pathlib import Path

import cv2


p=argparse.ArgumentParser()
p.add_argument("root",type=Path)
args=p.parse_args()
for episode in (0,7,27,56,63):
    cap=cv2.VideoCapture(str(args.root/f"demo-{episode}.mp4"))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert count>415 and round(cap.get(cv2.CAP_PROP_FPS))==50
    cells=[]
    for i in (0,50,94,129,168,193,225,257,290,325,387,count-1):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ok,frame=cap.read()
        assert ok
        frame=cv2.resize(frame,(320,160))
        cv2.putText(frame,f"{i/50:.2f} s",(4,15),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        cells.append(frame)
    cap.release()
    sheet=cv2.vconcat([cv2.hconcat(cells[i:i+3]) for i in range(0,12,3)])
    assert cv2.imwrite(str(args.root/f"review-{episode}.jpg"),sheet)
