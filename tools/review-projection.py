"""Overlay native image joints and actual evaluated rig joints at matched times."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--times", default="0,.5,1,2.233,8.77,10.13")
args = parser.parse_args()
if args.output.exists(): parser.error("Output must be a new file.")
report = json.loads(args.report.read_text(encoding="utf8"))
edges = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
capture = cv2.VideoCapture(str(args.video))
tiles = []
for time in map(float, args.times.split(",")):
    sample = min(report, key=lambda f: abs(f["time"] - time))
    capture.set(cv2.CAP_PROP_POS_MSEC, sample["sourceTime"] * 1000)
    ok, frame = capture.read()
    if not ok: raise ValueError("Could not read video frame")
    joints = {j["index"]: j for j in sample["joints"]}
    for field, color, thickness in (("native", (100, 255, 150), 3), ("rig", (255, 100, 240), 2)):
        for a, b in edges:
            if min(joints[a]["score"], joints[b]["score"]) >= .35:
                cv2.line(frame, tuple(round(v) for v in joints[a][field]), tuple(round(v) for v in joints[b][field]), color, thickness, cv2.LINE_AA)
    frame = cv2.resize(frame, (270, 480))
    cv2.putText(frame, f"{sample['time']:.2f}s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "green: image; pink: rig", (8, 465), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(frame)
capture.release()
with args.output.open("xb") as output:
    output.write(cv2.imencode(args.output.suffix, np.concatenate(tiles, axis=1))[1].tobytes())
print(args.output)
