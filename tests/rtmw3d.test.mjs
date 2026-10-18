import test from 'node:test';
import assert from 'node:assert/strict';
import {backproject,buildCapture,calibrate,fitDepth,FACE_EDGES} from '../backends/rtmw3d/motion.mjs';
import {estimateOrthographicTranslation} from '../root-motion.mjs';

function fixture() {
  const width=640,height=960,focal=width/(2*Math.tan(Math.PI/6));
  const local=Array.from({length:133},()=>[0,-.6,0]);
  Object.assign(local,{0:[0,-.7,-.09],1:[.04,-.73,-.07],2:[-.04,-.73,-.07],
    3:[.08,-.7,0],4:[-.08,-.7,0],5:[.2,-.5,0],6:[-.2,-.5,0],
    7:[.35,-.25,0],8:[-.35,-.25,0],9:[.4,0,-.05],10:[-.4,0,-.05],
    11:[.12,0,0],12:[-.12,0,0],13:[.12,.42,.06],14:[-.12,.42,.06],
    15:[.12,.84,0],16:[-.12,.84,0],17:[.12,.9,-.15],18:[.15,.9,-.15],19:[.12,.9,.04],
    20:[-.12,.9,-.15],21:[-.15,.9,-.15],22:[-.12,.9,.04]});
  for(const [start,wrist,sign] of [[91,9,1],[112,10,-1]]) {
    local[start]=[...local[wrist]];
    for(const [f,base] of [1,5,9,13,17].entries()) {
      for(let j=0;j<4;j++) local[start+base+j]=local[wrist].map((v,axis)=>v+
        (axis===0 ? sign*((f-2)*.017) : axis===1 ? .055+j*.025 : 0));
    }
  }
  const frames=Array.from({length:36},(_,i)=>({time:i/24,keypoints:local.map(([x,y,z])=>{
    const depth=3+i*.005;
    return [(x+i*.008)*focal/(depth+z)+width/2,y*focal/(depth+z)+height/2,z,.9];
  })}));
  return {format:'framecraft-rtmw3d-tracking',version:1,metadata:{width,height,fps:24,heightMeters:1.7,fov:60,engine:'rtmw3d-x'},frames};
}

test('backprojection respects perspective and metric relative depth',()=>{
  const t=fixture(), camera={width:640,height:960,focal:640/(2*Math.tan(Math.PI/6))};
  const p=backproject(t.frames[0].keypoints,3,camera);
  assert.ok(Math.abs(p[15][0]-.12)<1e-10);
  assert.ok(Math.abs(p[15][1]-.84)<1e-10);
  assert.ok(Math.abs(p[13][2]-3.06)<1e-10);
});

test('calibration/depth fitting retain real toward-camera and sideways travel',()=>{
  const t=fixture(),c=calibrate(t);
  const a=fitDepth(t.frames[0].keypoints,c),b=fitDepth(t.frames.at(-1).keypoints,c);
  assert.ok(a>2&&a<4);
  assert.ok(b-a>.12&&b-a<.24);
  const result=buildCapture(t,{pitch:0,cleanupStrength:0});
  assert.equal(result.frames.length,36);
  assert.equal(result.rootMotion.validFrames,36);
  assert.ok(result.frames.at(-1).root.position.x>.2);
  assert.ok(result.frames.at(-1).root.position.y>.1);
});

test('native face68 and hands are not fabricated as MediaPipe face478/expressions',()=>{
  const result=buildCapture(fixture(),{cleanupStrength:.25});
  assert.equal(result.faceTopology,'coco-wholebody-68');
  assert.equal(result.frames[0].face.world.length,68);
  assert.equal(result.frames[0].face.landmarks.length,68);
  assert.deepEqual(result.frames[0].face.blendshapes,{});
  assert.equal(result.frames[0].body.world.length,33);
  assert.equal(result.frames[0].hands[0].world.length,21);
  assert.ok(FACE_EDGES.every(edge=>edge.every(i=>i<68)));
  assert.equal(result.frames[0].body.world[23].visibility,.9);
});

test('image-constrained fitting retains per-sample camera centers independently of root cleanup',()=>{
  const t=fixture(),before=structuredClone(t);
  const result=buildCapture(t,{cleanupStrength:.5});
  assert.equal(result.metadata.poseFitting.method,'image-constrained-limbs-v1');
  const {camera}=result.metadata.calibration;
  const unfiltered=buildCapture(t,{cleanupStrength:0});
  for(let i=0;i<t.frames.length;i++) {
    const k=t.frames[i].keypoints,p=backproject(k,fitDepth(k,result.metadata.calibration),camera);
    for(const [a,axis] of ['x','y','z'].entries()) {
      assert.ok(Math.abs(result.frames[i].projection.cameraCenter[axis]-(p[11][a]+p[12][a])/2)<1e-10);
    }
    assert.deepEqual(result.frames[i].projection,unfiltered.frames[i].projection);
  }
  assert.deepEqual(t,before,'Native predictions stay untouched');
});

test('missing/low-confidence native detections stay missing, not invented',()=>{
  const t=fixture();t.frames[5].keypoints=null;
  t.frames[6].keypoints[5][3]=.1;
  const result=buildCapture(t);
  assert.equal(result.frames[5].body,null);
  assert.equal(result.frames[6].body,null);
  assert.equal(result.frames[5].root.status,'held');
});

test('invalid native data and timestamps are rejected before export',()=>{
  for(const mutate of [t=>t.frames[0].keypoints.pop(),t=>t.frames[0].keypoints[0][0]=NaN,
    t=>t.frames[1].time=0,t=>t.metadata.heightMeters=0,t=>t.metadata.fov=180]) {
    const t=fixture();mutate(t);assert.throws(()=>buildCapture(t));
  }
});

test('automatic workflow respects disabled root and foot stabilization',()=>{
  const result=buildCapture(fixture(),{rootEnabled:false,stabilize:false});
  assert.equal(result.rootMotion,undefined);
  assert.equal(result.frames[0].root,undefined);
  const rooted=buildCapture(fixture(),{stabilize:false});
  assert.equal(rooted.rootMotion.stabilize,false);
  assert.equal(buildCapture(fixture(),{cleanupStrength:0}).metadata.animationSmoothing.strength,0);
  assert.equal(buildCapture(fixture(),{cleanupStrength:.5}).metadata.animationSmoothing.strength,.5);
});

test('confident hand depth explosions are rejected, not made into giant fingers',()=>{
  const t=fixture(),before=structuredClone(t);
  // Bad first rest candidate, and a later occlusion after valid motion.
  t.frames[0].keypoints[114][2]+=1.2;
  t.frames[18].keypoints[99][2]-=.9;
  const result=buildCapture(t,{cleanupStrength:0});
  assert.deepEqual(result.frames[0].hands.map(h=>h.side),['Left']);
  assert.deepEqual(result.frames[18].hands.map(h=>h.side),['Right']);
  assert.equal(result.frames[1].hands.length,2);
  assert.equal(result.frames[0].body.world[20].visibility,0);
  assert.equal(result.frames[18].body.world[19].visibility,0);
  assert.equal(result.metadata.handQuality.accepted.Right,35);
  assert.equal(result.metadata.handQuality.accepted.Left,35);
  assert.deepEqual(t.frames[1],before.frames[1]);
  assert.equal(t.frames[0].keypoints[114][3],.9,'Raw predictions must not be mutated');
});

test('collapsed or individually low-confidence fingers are missing even with a high median score',()=>{
  const t=fixture();
  t.frames[2].keypoints[95]=[...t.frames[2].keypoints[94]];
  t.frames[3].keypoints[132][3]=.1;
  const result=buildCapture(t,{cleanupStrength:0});
  assert.equal(result.frames[2].hands.some(h=>h.side==='Left'),false);
  assert.equal(result.frames[3].hands.some(h=>h.side==='Right'),false);
  assert.equal(result.frames[4].hands.length,2);
});

test('one hand behind the camera cannot discard an otherwise valid body or opposite hand',()=>{
  const t=fixture();
  t.frames[7].keypoints[95][2]-=8;
  const result=buildCapture(t,{cleanupStrength:0});
  assert.ok(result.frames[7].body);
  assert.ok(result.frames[7].face);
  assert.deepEqual(result.frames[7].hands.map(h=>h.side),['Right']);
  assert.equal(result.rootMotion.validFrames,36);
});

test('temporal model body uses anatomical sides, a single scale and its own camera fit',()=>{
  const t=fixture(),calibration=calibrate(t);
  t.metadata.bodyEstimator={model:'motionagformer-b',layout:'h36m17'};
  for(const f of t.frames) {
    const p=backproject(f.keypoints,3,calibration.camera);
    const center=mean(p[11],p[12]);
    const model=Array.from({length:17},()=>[0,0,0]);
    for(const [a,b] of [[4,11],[1,12],[5,13],[2,14],[6,15],[3,16],[11,5],[14,6],[12,7],[15,8],[13,9],[16,10]]) model[a]=p[b].map((v,i)=>v-center[i]);
    model[8]=mean(model[11],model[14]); model[7]=model[8].map(v=>v*.5);
    model[9]=[0,-.6,0];model[10]=[0,-.75,0];
    f.temporalBody=model;
    f.body2d=f.keypoints.slice(0,17).map(p=>[p[0],p[1],p[3]]);
  }
  // This synthetic performer slides with a constant pose. Disable support-foot
  // locking to test the camera-derived travel, independently of contact fitting.
  const before=structuredClone(t), result=buildCapture(t,{cleanupStrength:0,pitch:0,stabilize:false});
  assert.deepEqual(t,before);
  assert.equal(result.metadata.poseFitting,undefined);
  assert.equal(result.metadata.bodyEstimator.rigFitting,'model-directions-v1');
  assert.equal(result.frames[0].body.pose17.length,17);
  assert.ok(result.frames[0].body.world[11].x>result.frames[0].body.world[12].x);
  assert.equal(result.rootMotion.cameraModel,'scaled-orthographic');
  assert.ok(result.frames.at(-1).root.position.x>.1);
  assert.equal(result.frames[0].projection.model,'scaled-orthographic');
  const cleaned=buildCapture(t,{cleanupStrength:.5,pitch:0,stabilize:false});
  for(let i=0;i<t.frames.length;i++) {
    assert.deepEqual(cleaned.frames[i].body,result.frames[i].body,'Cleanup must preserve the coherent learned body17/body33 pair');
    for(const [body,joint] of [[11,11],[12,14],[13,12],[14,15],[15,13],[16,16],[23,4],[24,1],[25,5],[26,2],[27,6],[28,3]]) {
      for(const axis of ['x','y','z']) assert.equal(cleaned.frames[i].body.world[body][axis],cleaned.frames[i].body.pose17[joint][axis]);
    }
  }
  t.frames[1].temporalBody=null;
  assert.equal(buildCapture(t).frames[1].body,null,'Never silently revert to framewise body');
  function mean(a,b){return a.map((v,i)=>(v+b[i])/2);}
});

test('temporal camera fit ignores local depth when estimating weak-perspective image scale',()=>{
  const t=buildCapture(fixture(),{cleanupStrength:0}),body=structuredClone(t.frames[0].body);
  const scale=400,tx=350,ty=380;
  body.world.forEach((p,i)=>{body.landmarks[i]={x:(scale*p.x+tx)/640,y:(scale*p.y+ty)/960,z:0,visibility:1};});
  const a=estimateOrthographicTranslation(body,{width:640,height:960});
  body.world.forEach((p,i)=>p.z=i%2?1:-1);
  const b=estimateOrthographicTranslation(body,{width:640,height:960});
  assert.deepEqual(a,b);
  assert.ok(Math.abs(a.pixelsPerMeter-scale)<1e-8);
  assert.ok(Math.abs(a.position.x-(tx-320)/scale)<1e-8);
});
