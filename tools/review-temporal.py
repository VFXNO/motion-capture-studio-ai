"""Review raw temporal-model body under a weak-perspective camera (not 3D GT)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path)
parser.add_argument("tracking", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error("Use a new output path")
data = json.loads(args.tracking.read_text(encoding="utf8"))
mapping = [(11,5),(12,7),(13,9),(14,6),(15,8),(16,10),(4,11),(5,13),(6,15),(1,12),(2,14),(3,16)]
edges = [(11,14),(11,12),(12,13),(14,15),(15,16),(11,4),(14,1),(4,1),(4,5),(5,6),(1,2),(2,3)]
capture = cv2.VideoCapture(str(args.video))
tiles = []
for time in (0,.5,2.233,5,8.77,11.8):
    frame = min(data["frames"], key=lambda f: abs(f["time"]-time))
    p = np.asarray(frame["temporalBody"])
    k = np.asarray(frame.get("body2d", frame["keypoints"]))
    x = p[[a for a,b in mapping],:2]
    y = k[[b for a,b in mapping],:2]
    a = x-x.mean(axis=0)
    b = y-y.mean(axis=0)
    scale = (a*b).sum()/(a*a).sum()
    pixel = (p[:,:2]-x.mean(axis=0))*scale+y.mean(axis=0)
    capture.set(cv2.CAP_PROP_POS_MSEC,frame["sourceTime"]*1000)
    ok,image = capture.read()
    if not ok: raise ValueError("Video decode failed")
    for a,b in edges:
        cv2.line(image,tuple(np.rint(pixel[a]).astype(int)),tuple(np.rint(pixel[b]).astype(int)),(255,100,240),2,cv2.LINE_AA)
    image = cv2.resize(image,(270,480))
    cv2.putText(image,f'{frame["time"]:.2f}s raw temporal / ortho',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    tiles.append(image)
capture.release()
args.output.open("xb").write(cv2.imencode(".png",np.concatenate(tiles,axis=1))[1].tobytes())
