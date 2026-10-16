"""Make a NEW contact sheet for visual 2D model review; not an accuracy metric."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("video",type=Path)
parser.add_argument("tracking",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
if args.output.exists(): parser.error("Output must be a new image.")
tracking=json.loads(args.tracking.read_text(encoding="utf8"))
body=[(3,5),(4,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(15,17),(16,20)]
hands=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
capture=cv2.VideoCapture(str(args.video))
tiles=[]
for index in np.linspace(0,len(tracking["frames"])-1,6,dtype=int):
    frame=tracking["frames"][index]
    capture.set(cv2.CAP_PROP_POS_MSEC,frame["sourceTime"]*1000)
    ok,image=capture.read()
    if not ok: raise ValueError("Video frame decode failed")
    keypoints=frame["keypoints"]
    if keypoints:
        for edges,color in [(body,(100,255,180))]+[([(a+start,b+start) for a,b in hands],(255,180,100)) for start in (91,112)]:
            for a,b in edges:
                if min(keypoints[a][3],keypoints[b][3])>.35:
                    cv2.line(image,tuple(round(v) for v in keypoints[a][:2]),tuple(round(v) for v in keypoints[b][:2]),color,3,cv2.LINE_AA)
        for x,y,z,score in keypoints[23:91]:
            if score>.35: cv2.circle(image,(round(x),round(y)),2,(210,140,255),-1)
    image=cv2.resize(image,(270,480))
    cv2.putText(image,f"RTMW3D {frame['time']:.2f}s",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    tiles.append(image)
capture.release()
sheet=np.concatenate(tiles,axis=1)
ok,encoded=cv2.imencode(args.output.suffix,sheet)
if not ok: raise ValueError("Image encode failed")
with args.output.open("xb") as handle: handle.write(encoded.tobytes())
print(args.output)
