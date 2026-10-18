import {test,expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const videoFixture=fileURLToPath(new URL('../fixtures/test.webm',import.meta.url));
const id='1234567890abcdef1234567890abcdef';

function nativeFixture() {
  const width=320,height=240,focal=width/(2*Math.tan(Math.PI/6));
  const local=Array.from({length:133},()=>[0,-.6,0]);
  for(const [sign,indices] of [[1,[3,5,7,9,11,13,15,17,18,19]],[-1,[4,6,8,10,12,14,16,20,21,22]]]) {
    [[.08,-.7,0],[.2,-.5,0],[.35,-.25,0],[.4,0,-.05],[.12,0,0],[.12,.42,.06],[.12,.84,0],[.12,.9,-.15],[.15,.9,-.15],[.12,.9,.04]]
      .forEach((p,i)=>{local[indices[i]]=[p[0]*sign,p[1],p[2]];});
  }
  for(const [start,wrist,sign] of [[91,9,1],[112,10,-1]]) {
    local[start]=[...local[wrist]];
    for(const [f,base] of [1,5,9,13,17].entries()) {
      for(let j=0;j<4;j++) local[start+base+j]=local[wrist].map((v,axis)=>v+
        (axis===0 ? sign*(f-2)*.017 : axis===1 ? .055+j*.025 : 0));
    }
  }
  const model=Array.from({length:17},()=>[0,0,0]);
  for(const [a,b] of [[4,11],[1,12],[5,13],[2,14],[6,15],[3,16],[11,5],[14,6],[12,7],[15,8],[13,9],[16,10]]) model[a]=local[b];
  model[7]=[0,-.25,0]; model[8]=[0,-.5,0]; model[9]=[0,-.6,0]; model[10]=[0,-.75,0];
  return {format:'framecraft-rtmw3d-tracking',version:1,
    metadata:{width,height,fps:24,heightMeters:1.7,fov:60,engine:'rtmw3d-x',bodyEstimator:{model:'motionagformer-b',layout:'h36m17'}},
    frames:Array.from({length:24},(_,i)=>{
      const keypoints=local.map(([x,y,z])=>[(x+i*.008)*focal/(3+i*.005+z)+width/2,y*focal/(3+i*.005+z)+height/2,z,.9]);
      return {time:i/24,keypoints,temporalBody:model,body2d:keypoints.slice(0,17).map(p=>[p[0],p[1],p[3]])};
    })};
}

async function mockLocal(page,{fail=false,wait=false}={}) {
  let stage='uploading',polls=0,exported=null,options=null,workerRequests=0,cancelled=false;
  await page.route('**/tracker.worker.mjs',route=>{workerRequests++;return route.abort();});
  await page.route('**/api/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    const send=body=>route.fulfill({contentType:'application/json',body:JSON.stringify(body)});
    if(path==='/api/session') return send({token:'f'.repeat(43),installed:false,automaticSetup:true,blenderAvailable:true});
    if(path==='/api/jobs') {options=request.postDataJSON();return send({id,state:'uploading'});}
    expect(request.headers()['x-framecraft-token']).toBe('f'.repeat(43));
    if(request.method()==='DELETE') {cancelled=true;stage='cancelled';return send({id,state:stage});}
    if(path.endsWith('/video')) {expect(request.method()).toBe('PUT');stage='preparing';return send({id,state:stage});}
    if(path.endsWith('/tracking')) return send(nativeFixture());
    if(path.endsWith('/export')) {exported=request.postDataJSON();stage='complete';return send({id,state:stage});}
    if(path.endsWith('/capture')) return send(exported);
    if(path.endsWith('/blend')) return route.fulfill({contentType:'application/octet-stream',body:Buffer.from('BLENDER mock scene')});
    if(fail) stage='failed';
    else if(!wait&&stage==='preparing'&&polls++>0) stage='tracking-ready';
    return send({id,state:stage,message:fail?'Inference failed for testing':'Automatic model setup and local inference',progress:.1,
      blendReady:stage==='complete',captureReady:stage==='complete'});
  });
  return {get options(){return options;},get exported(){return exported;},get workerRequests(){return workerRequests;},get cancelled(){return cancelled;}};
}

test('one click runs automatic setup/capture/export and downloads a Blender scene',async({page})=>{
  const mock=await mockLocal(page);
  await page.goto('/');
  await expect(page.locator('#captureEngine')).toHaveValue('rtmw3d');
  await expect(page.locator('#engineNote')).toContainText('installs everything automatically');
  await page.locator('#fileInput').setInputFiles(videoFixture);
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Your capture is ready');
  await expect(page.locator('#statusText')).toContainText('no import script needed');
  await expect(page.locator('#rootPreviewPanel')).toBeVisible();
  expect(mock.options.makeBlend).toBe(true);
  expect(mock.exported.metadata.engine).toBe('rtmw3d-x');
  expect(mock.exported.metadata.bodyEstimator.model).toBe('motionagformer-b');
  expect(mock.exported.metadata.poseFitting).toBeUndefined();
  expect(mock.exported.frames[0].projection.model).toBe('scaled-orthographic');
  expect(mock.exported.frames[0].projection.cameraCenter.z).toBeGreaterThan(.1);
  expect(mock.exported.frames[0].face.world).toHaveLength(68);
  expect(mock.exported.frames[0].hands).toHaveLength(2);
  expect(mock.workerRequests).toBe(0);
  const downloadEvent=page.waitForEvent('download');
  await page.locator('#blendButton').click();
  const download=await downloadEvent;
  expect(download.suggestedFilename()).toBe('test.blend');
  expect((await readFile(await download.path(),'utf8'))).toContain('BLENDER');
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});

test('local cancellation releases the job and offers no partial JSON or Blender download',async({page})=>{
  const mock=await mockLocal(page,{wait:true});
  await page.goto('/');
  await page.locator('#fileInput').setInputFiles(videoFixture);
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusText')).toContainText('Automatic model setup');
  await page.locator('#cancelButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Capture cancelled');
  await expect(page.locator('#exportButton')).toBeDisabled();
  await expect(page.locator('#blendButton')).toBeHidden();
  expect(mock.cancelled).toBe(true);
  expect(mock.workerRequests).toBe(0);
});

test('local failure is explicit and never silently invokes MediaPipe',async({page})=>{
  const mock=await mockLocal(page,{fail:true});
  await page.goto('/');
  await page.locator('#fileInput').setInputFiles(videoFixture);
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Capture could not finish');
  await expect(page.locator('#statusText')).toContainText('No MediaPipe fallback');
  await expect(page.locator('#exportButton')).toBeDisabled();
  expect(mock.workerRequests).toBe(0);
});

test('real RTMW3D video to downloadable Blender scene without manual commands',async({page})=>{
  test.skip(!process.env.REAL_RTM_VIDEO,'Set REAL_RTM_VIDEO to a local video for actual inference and Blender.');
  test.setTimeout(240000);
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#engineNote')).toContainText('Installed on this PC');
  await page.locator('#fileInput').setInputFiles(process.env.REAL_RTM_VIDEO);
  const captured=page.waitForResponse(response=>/\/api\/jobs\/[a-f0-9]{32}\/capture$/.test(new URL(response.url()).pathname));
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Your capture is ready',{timeout:210000});
  const response=await captured;
  const capture=await response.json();
  expect(capture.metadata.engine).toBe('rtmw3d-x');
  expect(capture.metadata.bodyEstimator.model).toBe('motionagformer-b');
  expect(capture.metadata.bodyEstimator.bodyDetector).toBe('rtmpose-x-body7');
  expect(capture.metadata.poseFitting).toBeUndefined();
  expect(capture.frames.find(f=>f.body).projection.model).toBe('scaled-orthographic');
  expect(capture.frames.length).toBeGreaterThan(2);
  expect(capture.rootMotion.enabled).toBe(true);
  await expect(page.locator('#blendButton')).toBeVisible();
  const event=page.waitForEvent('download');
  await page.locator('#blendButton').click();
  const download=await event;
  const contents=await readFile(await download.path());
  expect(contents.length).toBeGreaterThan(10000);
  // Blender 5.x can write Zstandard-compressed scenes by default.
  expect(contents.subarray(0,7).toString()==='BLENDER'||contents.subarray(0,4).toString('hex')==='28b52ffd').toBe(true);
  const jobID=response.url().match(/jobs\/([^/]+)/)[1];
  const saved=await readFile(fileURLToPath(new URL(`../../.rtmw3d-runs/${jobID}/motion.blend`,import.meta.url)));
  expect(contents.equals(saved)).toBe(true);
  expect(errors).toEqual([]);
  console.log('AUTOMATIC_RTMW3D_OK',JSON.stringify({job:jobID,samples:capture.frames.length,blendBytes:contents.length}));
  await page.screenshot({path:'test-results/automatic-rtmw3d.png',fullPage:true});
});
