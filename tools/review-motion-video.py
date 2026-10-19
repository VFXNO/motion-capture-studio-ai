"""Write a new synchronized source/rig projection comparison video; no audio."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from backends.rtmw3d.capture import sample_video

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video",type=Path)
parser.add_argument("report",type=Path)
parser.add_argument("output",type=Path)
parser.add_argument("--fps",type=int,default=30)
parser.add_argument("--before",type=Path,help="Optional earlier rig projection report for a three-panel comparison")
args = parser.parse_args()
if args.output.exists(): parser.error("Use a new output filename")
report=json.loads(args.report.read_text(encoding="utf8"))
before=json.loads(args.before.read_text(encoding="utf8")) if args.before else None
edges=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
samples=sample_video(args.video,args.fps)
info=next(samples)
width,height=360,640
writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*"mp4v"),args.fps,(width*(3 if before else 2),height))
if not writer.isOpened(): raise RuntimeError("Could not create comparison video")
try:
    for index,(time,source,image) in enumerate(samples):
        left=cv2.resize(image,(width,height))
        tiles=[(left,"Source video")]
        comparisons=[(before,"Previous temporal rig"),(report,"Corrected rig / estimated camera")] if before else [(report,"Temporal rig / estimated camera")]
        for data,label in comparisons:
            overlay=image.copy()
            frame=min(data,key=lambda f:abs(f["time"]-time))
            if abs(frame["time"]-time)<.5/args.fps:
                points={j["index"]:tuple(round(v) for v in j["rig"]) for j in frame["joints"]}
                for a,b in edges:
                    cv2.line(overlay,points[a],points[b],(70,235,255),3,cv2.LINE_AA)
                for point in points.values(): cv2.circle(overlay,point,4,(255,80,200),-1)
            tiles.append((cv2.resize(overlay,(width,height)),label))
        for tile,label in tiles:
            cv2.rectangle(tile,(0,0),(width,48),(20,20,20),-1)
            cv2.putText(tile,label,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(tile,f'Frame {index+1} | {time:.2f}s',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        writer.write(np.concatenate([tile for tile,label in tiles],axis=1))
finally:
    writer.release()
print(args.output)
