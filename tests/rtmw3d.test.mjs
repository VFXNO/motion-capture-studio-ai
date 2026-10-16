import test from 'node:test';
import assert from 'node:assert/strict';
import {backproject,buildCapture,calibrate,fitDepth,FACE_EDGES} from '../backends/rtmw3d/motion.mjs';

function fixture() {
  const width=640,height=960,focal=width/(2*Math.tan(Math.PI/6));
  const local=Array.from({length:133},()=>[0,-.6,0]);
  Object.assign(local,{0:[0,-.7,-.09],1:[.04,-.73,-.07],2:[-.04,-.73,-.07],
    3:[.08,-.7,0],4:[-.08,-.7,0],5:[.2,-.5,0],6:[-.2,-.5,0],
    7:[.35,-.25,0],8:[-.35,-.25,0],9:[.4,0,-.05],10:[-.4,0,-.05],
    11:[.12,0,0],12:[-.12,0,0],13:[.12,.42,.06],14:[-.12,.42,.06],
    15:[.12,.84,0],16:[-.12,.84,0],17:[.12,.9,-.15],18:[.15,.9,-.15],19:[.12,.9,.04],
    20:[-.12,.9,-.15],21:[-.15,.9,-.15],22:[-.12,.9,.04]});
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
});
