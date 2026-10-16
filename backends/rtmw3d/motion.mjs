// RTMW3D predicts image XY and relative metric depth, NOT metric XYZ or world
// travel. Backproject with estimated intrinsics and actor-height calibration.
// Keep this assumption explicit; never label crop pixels as meters.
import {rebuildCaptureRoot} from '../../root-motion.mjs';

const pairs = [[3,5],[4,6],[5,11],[6,12],[11,13],[12,14],[13,15],[14,16]];
const sides = [[3,5,11,13,15], [4,6,12,14,16]];
const median = values => [...values].sort((a,b) => a-b)[Math.floor(values.length/2)];
const distance = (a,b) => Math.hypot(...a.map((v,i) => v-b[i]));
const mean = (a,b) => a.map((v,i) => (v+b[i])/2);
const visibility = value => Math.max(0, Math.min(1, value));

export function backproject(keypoints, depth, {width, height, focal}) {
  const zCenter = (keypoints[11][2] + keypoints[12][2])/2;
  return keypoints.map(([x,y,z]) => {
    const absoluteZ = depth + z-zCenter;
    return [(x-width/2)/focal*absoluteZ, (y-height/2)/focal*absoluteZ, absoluteZ];
  });
}

export function calibrate(tracking) {
  const {width,height,fov,heightMeters} = tracking.metadata;
  const camera = {width,height,focal: width/(2*Math.tan(fov*Math.PI/360))};
  const usable = tracking.frames.filter(frame => frame.keypoints &&
    [3,4,5,6,11,12,13,14,15,16].every(i => frame.keypoints[i][3] >= .35));
  if (!usable.length) throw new Error('No reliable full-body samples for actor-height calibration.');
  // Ear-to-ankle chain plus a 14cm head/sole allowance. Both height and this
  // anthropometric assumption are approximate, not a recovered body model.
  const chainTarget = heightMeters-.14;
  const chainLength = positions => sides.reduce((total, indices) => total +
    indices.slice(1).reduce((sum, index, i) => sum+distance(positions[indices[i]],positions[index]),0),0)/2;
  const initial = usable.map(frame => {
    let lo=.5, hi=40;
    for (let i=0;i<36;i++) {
      const depth=(lo+hi)/2;
      if (chainLength(backproject(frame.keypoints,depth,camera)) < chainTarget) lo=depth;
      else hi=depth;
    }
    return {keypoints:frame.keypoints,depth:(lo+hi)/2};
  }).filter(frame => frame.depth > .55 && frame.depth < 39);
  if (!initial.length) throw new Error('Actor-height fit failed; check height/FOV and visible feet.');
  const lengths = pairs.map(([a,b]) => median(initial.map(frame => {
    const p=backproject(frame.keypoints,frame.depth,camera);
    return distance(p[a],p[b]);
  })));
  return {camera,lengths,medianDepth:median(initial.map(f=>f.depth)),method:'height-calibrated-ray-depth-v1',
    assumptions:'Estimated FOV, actor height and 14cm head/sole allowance; fixed camera. No SLAM, learned root trajectory, or bone twist.'};
}

export function fitDepth(keypoints, calibration) {
  const {camera,lengths,medianDepth} = calibration;
  function cost(depth) {
    const points=backproject(keypoints,depth,camera);
    let total=0, weight=0;
    pairs.forEach(([a,b],i)=>{
      const confidence=Math.min(keypoints[a][3],keypoints[b][3]);
      if (confidence < .35) return;
      const error=(distance(points[a],points[b])-lengths[i])/Math.max(.1,lengths[i]);
      // Huber loss: one occluded limb must not control global scale/depth.
      total+=confidence*(Math.abs(error)<.15 ? error*error : .3*Math.abs(error)-.0225);
      weight+=confidence;
    });
    return weight>1 ? total/weight : Infinity;
  }
  let lo=Math.max(.5,medianDepth/3), hi=Math.min(40,medianDepth*3);
  for(let i=0;i<44;i++) {
    const a=lo+(hi-lo)/3,b=hi-(hi-lo)/3;
    if(cost(a)<cost(b)) hi=b; else lo=a;
  }
  const depth=(lo+hi)/2;
  return cost(depth)<.12 ? depth : null;
}

export const FACE_EDGES = [];
for(const [start,end,closed] of [[0,16,false],[17,21,false],[22,26,false],[27,30,false],
  [31,35,false],[36,41,true],[42,47,true],[48,59,true],[60,67,true]]) {
  for(let i=start;i<end;i++) FACE_EDGES.push([i,i+1]);
  if(closed) FACE_EDGES.push([end,start]);
}

export function buildCapture(tracking, {pitch=10,cleanupStrength=.25,stabilize=true,rootEnabled=true}={}) {
  const m=tracking?.metadata;
  if(tracking?.format!=='framecraft-rtmw3d-tracking'||tracking.version!==1||!m||
    ![m.width,m.height,m.fps,m.heightMeters,m.fov].every(Number.isFinite)||
    m.width<=0||m.height<=0||m.width>32768||m.height>32768||m.fps<=0||m.fps>30||
    m.heightMeters<1||m.heightMeters>2.5||m.fov<30||m.fov>100||
    !Array.isArray(tracking.frames)||!tracking.frames.length||tracking.frames.length>1800) throw new Error('Invalid native RTMW3D tracking capture.');
  let previous=-1;
  for(const frame of tracking.frames) {
    if(!Number.isFinite(frame.time)||frame.time<0||frame.time<=previous) throw new Error('Invalid native timestamps.');
    previous=frame.time;
    if(frame.keypoints!==null && (!Array.isArray(frame.keypoints)||frame.keypoints.length!==133||
      !frame.keypoints.every(p=>Array.isArray(p)&&p.length===4&&p.every(Number.isFinite)&&p[3]>=0))) throw new Error('Expected 133 finite native keypoints.');
  }
  const calibration=calibrate(tracking);
  const frames=tracking.frames.map(frame=>{
    const base={time:frame.time,body:null,hands:[],face:null};
    const k=frame.keypoints;
    if(!k||[5,6,11,12].some(i=>k[i][3]<.35)) return base;
    const depth=fitDepth(k,calibration);
    if(depth===null) return base;
    const cameraPoints=backproject(k,depth,calibration.camera);
    if(cameraPoints.some(p=>p[2]<.1)) return base;
    const center=mean(cameraPoints[11],cameraPoints[12]);
    const world=cameraPoints.map((p,i)=>({x:p[0]-center[0],y:p[1]-center[1],z:p[2]-center[2],visibility:visibility(k[i][3])}));
    const image=k.map((p,i)=>({x:p[0]/m.width,y:p[1]/m.height,z:world[i].z/m.width,visibility:visibility(p[3])}));
    // Explicit compatibility mapping, not MediaPipe inference. Eye corners,
    // mouth corners and fingertips come from the native face/hand estimates.
    const indices=[0,65,66,68,59,60,62,3,4,77,71,5,6,7,8,9,10,111,132,99,120,95,116,11,12,13,14,15,16,19,22,17,20];
    base.body={world:indices.map(i=>({...world[i]})),landmarks:indices.map(i=>({...image[i]}))};
    for(const [side,start] of [['Left',91],['Right',112]]) {
      const score=median(k.slice(start,start+21).map(p=>p[3]));
      if(score>=.35) base.hands.push({side,score:visibility(score),world:world.slice(start,start+21),landmarks:image.slice(start,start+21)});
    }
    if(median(k.slice(23,91).map(p=>p[3]))>=.35) base.face={topology:'coco-wholebody-68',
      world:world.slice(23,91),landmarks:image.slice(23,91),blendshapes:{}};
    return base;
  });
  const capture={format:'framecraft-mocap',version:1,coordinateSystem:'Camera x right, y down, z away; estimated meters; body hips centered. RTMW3D mapped to body33/hand21; native metric face68.',
    metadata:{...m,calibration,notes:'RTMW3D-X inference. Approximate actor-height/FOV backprojection and grounded root fit. Native face68 has no expression scores. Body33 is a compatibility mapping; no MediaPipe inference used.'},
    faceTopology:'coco-wholebody-68',faceEdges:FACE_EDGES,frames};
  const result = rebuildCaptureRoot(capture,{fov:m.fov,pitch,stabilize,cleanupStrength});
  if (!rootEnabled) {
    result.frames = result.frames.map(({root, ...frame}) => frame);
    delete result.rootMotion;
  }
  return result;
}
