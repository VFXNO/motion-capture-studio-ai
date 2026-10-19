"""Compare body detections and raw neural arm projections at failure timestamps."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('video',type=Path)
parser.add_argument('tracking',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--pose-field',default='temporalBody',choices=('temporalBody','refinedBody'))
args=parser.parse_args()
if args.output.exists(): parser.error('Use a new output filename')
data=json.loads(args.tracking.read_text(encoding='utf8'))
mapping=[(11,5),(12,7),(13,9),(14,6),(15,8),(16,10),(4,11),(5,13),(6,15),(1,12),(2,14),(3,16)]
video=cv2.VideoCapture(str(args.video))
rows=[]
for time in (2.233,2.3,10.133,10.967,12.167):
    f=min(data['frames'],key=lambda x:abs(x['time']-time))
    p=np.asarray(f[args.pose_field]); k=np.asarray(f['body2d']); native=np.asarray(f['keypoints'])
    x=p[[a for a,b in mapping],:2]; y=k[[b for a,b in mapping],:2]
    a=x-x.mean(0); b=y-y.mean(0); scale=(a*b).sum()/(a*a).sum()
    pixel=(p[:,:2]-x.mean(0))*scale+y.mean(0)
    video.set(cv2.CAP_PROP_POS_MSEC,f['sourceTime']*1000)
    ok,image=video.read()
    if not ok: raise ValueError('Decode failed')
    tiles=[]
    for label,points,chains in [('Video',None,()),('RTMPose-X',k,((5,7,9),(6,8,10))),('RTMW3D',native,((5,7,9),(6,8,10))),('Temporal model',pixel,((11,12,13),(14,15,16)))]:
        tile=image.copy()
        for chain,color in zip(chains,((0,230,255),(255,130,70))):
            for start,end in zip(chain,chain[1:]): cv2.line(tile,tuple(np.rint(points[start,:2]).astype(int)),tuple(np.rint(points[end,:2]).astype(int)),color,3,cv2.LINE_AA)
            for j in chain:
                pos=tuple(np.rint(points[j,:2]).astype(int)); cv2.circle(tile,pos,5,color,-1)
                cv2.putText(tile,str(j),pos,cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        tile=cv2.resize(tile,(270,480))
        cv2.rectangle(tile,(0,0),(270,27),(0,0,0),-1)
        cv2.putText(tile,f'{f["time"]:.3f}s {label}',(4,19),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
        tiles.append(tile)
    rows.append(np.concatenate(tiles,axis=1))
video.release()
args.output.open('xb').write(cv2.imencode('.png',np.concatenate(rows,axis=0))[1].tobytes())
