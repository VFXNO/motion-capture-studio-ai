// RTMW3D predicts image XY and relative metric depth, NOT metric XYZ or world
// travel. Backproject with estimated intrinsics and actor-height calibration.
// Keep this assumption explicit; never label crop pixels as meters.
import {rebuildCaptureRoot,estimateCameraTranslation} from '../../root-motion.mjs';

const pairs = [[3,5],[4,6],[5,11],[6,12],[11,13],[12,14],[13,15],[14,16]];
const sides = [[3,5,11,13,15], [4,6,12,14,16]];
const median = values => [...values].sort((a,b) => a-b)[Math.floor(values.length/2)];
const distance = (a,b) => Math.hypot(...a.map((v,i) => v-b[i]));
const mean = (a,b) => a.map((v,i) => (v+b[i])/2);
const visibility = value => Math.max(0, Math.min(1, value));
const handStarts = [1,5,9,13,17];
const temporalMap = [[23,4],[24,1],[25,5],[26,2],[27,6],[28,3],
  [11,11],[12,14],[13,12],[14,15],[15,13],[16,16]];

function temporalScale(tracking) {
  const chain = p => distance(p[10],p[8])+distance(p[8],p[0])+
    (distance(p[0],p[1])+distance(p[1],p[2])+distance(p[2],p[3])+
     distance(p[0],p[4])+distance(p[4],p[5])+distance(p[5],p[6]))/2;
  const lengths=tracking.frames.filter(f=>f.temporalBody).map(f=>chain(f.temporalBody));
  const typical=median(lengths);
  if(!Number.isFinite(typical)||typical<.1) throw new Error('Temporal model did not produce a usable body.');
  return (tracking.metadata.heightMeters-.14)/typical;
}

function useTemporalBody(frame, pose, scale) {
  const xyz = p => [p.x,p.y,p.z];
  const point = (p,visibility=1) => ({x:p[0],y:p[1],z:p[2],visibility});
  const modelCenter=mean(pose[1],pose[4]);
  const p=pose.map(j=>j.map((value,axis)=>(value-modelCenter[axis])*scale));
  const old=frame.body.world,world=old.map(j=>({...j}));
  for(const [body,joint] of temporalMap) world[body]=point(p[joint],frame.body.landmarks[body].visibility);
  // MotionAGFormer predicts body17, not face/feet/fingers. Keep those local
  // details explicitly sourced from RTMW3D and attach them at the new anchors.
  const head=mean(xyz(old[7]),xyz(old[8]));
  const headAnchor=mean(p[9],p[10]);
  for(let i=0;i<=10;i++) world[i]=point(xyz(old[i]).map((v,a)=>v-head[a]+headAnchor[a]),old[i].visibility);
  for(const [ankle,details] of [[27,[29,31]],[28,[30,32]],[15,[17,19,21]],[16,[18,20,22]]]) {
    for(const index of details) {
      const delta=xyz(old[index]).map((v,a)=>v-xyz(old[ankle])[a]);
      const limit=ankle>=27?.25:.16,ratio=Math.min(1,limit/Math.max(1e-8,Math.hypot(...delta)));
      world[index]=point(delta.map((v,a)=>v*ratio+xyz(world[ankle])[a]),old[index].visibility);
    }
  }
  frame.body={...frame.body,world,pose17:p.map(j=>point(j))};
  frame.hands=frame.hands.map(hand=>{
    const wrist=xyz(hand.world[0]),anchor=xyz(world[hand.side==='Left'?15:16]);
    return {...hand,world:hand.world.map(j=>point(xyz(j).map((v,a)=>v-wrist[a]+anchor[a]),j.visibility))};
  });
  if(frame.face) frame.face={...frame.face,world:frame.face.world.map(j=>point(xyz(j).map((v,a)=>v-head[a]+headAnchor[a]),j.visibility))};
  delete frame.projection;
}

function plausibleHand(points, scale) {
  // A confident 2D detection can still have a wildly wrong depth. Do not feed
  // meter-long phalanges to cleanup or to the Blender rest-pose estimator.
  const xyz = p => [p.x,p.y,p.z];
  if(points.some(p => p.visibility < .35)) return false;
  for(const start of handStarts) {
    const palm = distance(xyz(points[0]),xyz(points[start]));
    if(palm < .005*scale || palm > .16*scale) return false;
    for(let j=start;j<start+3;j++) {
      const length=distance(xyz(points[j]),xyz(points[j+1]));
      if(length < .002*scale || length > .085*scale) return false;
    }
  }
  return points.every(p => distance(xyz(points[0]),xyz(p)) <= .28*scale);
}

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
  const temporal=m?.bodyEstimator?.model==='motionagformer-b';
  const refined=m?.bodyEstimator?.refinement?.method==='observed-arms-neural-depth-v1';
  if(m?.bodyEstimator?.refinement && !refined) throw new Error('Unsupported temporal arm refinement.');
  if(m?.bodyEstimator && !temporal) throw new Error('Unsupported temporal body model.');
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
    if(frame.temporalBody && (!temporal || !Array.isArray(frame.temporalBody)||frame.temporalBody.length!==17||
      !frame.temporalBody.every(p=>Array.isArray(p)&&p.length===3&&p.every(v=>Number.isFinite(v)&&Math.abs(v)<100)))) throw new Error('Invalid temporal body prediction.');
    if(frame.body2d && (!Array.isArray(frame.body2d)||frame.body2d.length!==17||!frame.body2d.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)&&p[2]>=0&&p[2]<=1))) throw new Error('Invalid dedicated body detections.');
    if(frame.refinedBody && (!temporal||!refined||!frame.temporalBody||!Array.isArray(frame.refinedBody)||frame.refinedBody.length!==17||
      !frame.refinedBody.every(p=>Array.isArray(p)&&p.length===3&&p.every(v=>Number.isFinite(v)&&Math.abs(v)<100)))) throw new Error('Invalid refined arm prediction.');
    if(refined&&frame.temporalBody&&!frame.refinedBody) throw new Error('Missing refined body; refusing mixed raw/fitted output.');
  }
  const calibration=calibrate(tracking);
  const modelScale=temporal?temporalScale(tracking):null;
  const frames=tracking.frames.map(frame=>{
    const base={time:frame.time,body:null,hands:[],face:null};
    const k=frame.keypoints;
    if(!k||[5,6,11,12].some(i=>(frame.body2d?.[i]?.[2]??k[i][3])<.35)) return base;
    if(temporal&&!frame.temporalBody) return base;
    const depth=fitDepth(k,calibration) ?? (temporal?calibration.medianDepth:null);
    if(depth===null) return base;
    const cameraPoints=backproject(k,depth,calibration.camera);
    if(!temporal && cameraPoints.slice(0,23).some(p=>p[2]<.1)) return base;
    const center=mean(cameraPoints[11],cameraPoints[12]);
    base.projection={cameraCenter:{x:center[0],y:center[1],z:center[2]}};
    const world=cameraPoints.map((p,i)=>({x:p[0]-center[0],y:p[1]-center[1],z:p[2]-center[2],visibility:visibility(k[i][3])}));
    const image=k.map((p,i)=>({x:p[0]/m.width,y:p[1]/m.height,z:world[i].z/m.width,visibility:visibility(p[3])}));
    if(temporal&&frame.body2d) for(let i=0;i<17;i++) {
      image[i]={...image[i],x:frame.body2d[i][0]/m.width,y:frame.body2d[i][1]/m.height,visibility:frame.body2d[i][2]};
    }
    // Explicit compatibility mapping, not MediaPipe inference. Eye corners,
    // mouth corners and fingertips come from the native face/hand estimates.
    const indices=[0,65,66,68,59,60,62,3,4,77,71,5,6,7,8,9,10,111,132,99,120,95,116,11,12,13,14,15,16,19,22,17,20];
    base.body={world:indices.map(i=>({...world[i]})),landmarks:indices.map(i=>({...image[i]}))};
    for(const [side,start,wrist,bodyTips] of [['Left',91,9,[17,19,21]],['Right',112,10,[18,20,22]]]) {
      const score=median(k.slice(start,start+21).map(p=>p[3]));
      const points=world.slice(start,start+21),scale=m.heightMeters/1.7;
      const wristGap=distance(cameraPoints[start],cameraPoints[wrist]);
      if(score>=.35 && wristGap<=.12*scale && cameraPoints.slice(start,start+21).every(p=>p[2]>=.1) && plausibleHand(points,scale)) {
        base.hands.push({side,score:visibility(score),world:points,landmarks:image.slice(start,start+21)});
      } else {
        // These body33 compatibility slots are also sourced from that hand.
        // Preserve their predictions but mark them unusable; do not invent tips.
        for(const index of bodyTips) {
          base.body.world[index].visibility=0;
          base.body.landmarks[index].visibility=0;
        }
      }
    }
    if(median(k.slice(23,91).map(p=>p[3]))>=.35 && cameraPoints.slice(23,91).every(p=>p[2]>=.1)) base.face={topology:'coco-wholebody-68',
      world:world.slice(23,91),landmarks:image.slice(23,91),blendshapes:{}};
    if(temporal) useTemporalBody(base,refined?frame.refinedBody:frame.temporalBody,modelScale);
    return base;
  });
  const capture={format:'framecraft-mocap',version:1,coordinateSystem:'Camera x right, y down, z away; estimated meters; body hips centered. RTMW3D mapped to body33/hand21; native metric face68.',
    metadata:{...m,calibration,poseFitting:{method:'image-constrained-limbs-v1',
      assumptions:'Visible joint image rays constrain limbs; depth branches use weak model and temporal priors. Monocular depth remains ambiguous.'},
      animationSmoothing:{method:'centered-targets-and-quaternion-reacquisition-v1',strength:cleanupStrength},handQuality:{method:'confidence-and-metric-geometry-v1',
      policy:'Reject implausible hand depth/size before cleanup; missing fingers hold their last pose in Blender. Raw native predictions are preserved separately.',
      accepted:Object.fromEntries(['Left','Right'].map(side=>[side,frames.filter(f=>f.hands.some(h=>h.side===side)).length]))},
      notes:'RTMW3D-X inference. Approximate actor-height/FOV backprojection and grounded root fit. Native face68 has no expression scores. Body33 is a compatibility mapping; no MediaPipe inference used.'},
    faceTopology:'coco-wholebody-68',faceEdges:FACE_EDGES,frames};
  if(temporal) {
    delete capture.metadata.poseFitting;
    capture.metadata.bodyEstimator={...m.bodyEstimator,scale:modelScale,rigFitting:'model-directions-v1'};
    capture.metadata.notes='MotionAGFormer-B temporal body from RTMW3D image detections; native RTMW3D hands/face/foot details. Approximate height calibration and ground travel. No image-ray depth-branch fitting.';
  }
  const result = rebuildCaptureRoot(capture,{fov:m.fov,pitch,stabilize,cleanupStrength});
  if(temporal) {
    result.frames = result.frames.map(frame=>{
      const solved=frame.body?estimateCameraTranslation(frame.body,{width:m.width,height:m.height,fov:m.fov}):null;
      return solved?{...frame,projection:{cameraCenter:solved.position,model:solved.cameraModel,pixelsPerMeter:solved.pixelsPerMeter}}:frame;
    });
  }
  if (!rootEnabled) {
    result.frames = result.frames.map(({root, ...frame}) => frame);
    delete result.rootMotion;
  }
  return result;
}
