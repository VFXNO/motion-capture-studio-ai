import {test, expect} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const fixture = fileURLToPath(new URL('../fixtures/test.webm', import.meta.url));
const fakeWorker = `
self.onmessage = async ({data}) => {
  if (data.type === 'init') {
    self.postMessage({id: data.id, result: {faceEdges: [[0, 1]]}});
    return;
  }
  data.bitmap.close();
  const points = count => Array.from({length: count}, (_, i) => ({x: 0.3 + i / 2000, y: 0.4 + i / 2000, z: 0, visibility: 1}));
  self.postMessage({id: data.id, result: {time: data.time,
    body: {landmarks: points(33), world: points(33)},
    hands: [{side: 'Left', score: 0.9, landmarks: points(21), world: points(21)}],
    face: {landmarks: points(478), blendshapes: {eyeBlinkLeft: 0.5}, matrix: []}}});
};`;

const fakeGroundWorker = fakeWorker.replace('body: {landmarks: points(33), world: points(33)},', 'body: makeBody(data.time),')
  .replace('data.bitmap.close();', `data.bitmap.close();
  function makeBody(time) {
    const world = Array.from({length: 33}, () => ({x: 0, y: -.65, z: 0}));
    for (const [side, indices] of [[-1,[11,13,23,25,27,29,31]], [1,[12,14,24,26,28,30,32]]]) {
      const values = [[side*.22,-.5,0],[side*.4,-.2,.02],[side*.11,0,0],[side*.11,.5,.03],
        [side*.11,.9,0],[side*.11,1,-.04],[side*.11,1,.16]];
      indices.forEach((joint,i) => { world[joint] = {x:values[i][0],y:values[i][1],z:values[i][2]}; });
    }
    const focal = 320 / (2 * Math.tan(Math.PI / 6));
    const landmarks = world.map(p => ({x:.5+(p.x+time*.4)/(p.z+6+time*.15)*focal/320,
      y:.5+(p.y+1)/(p.z+6+time*.15)*focal/240,z:0,visibility:1}));
    return {world,landmarks};
  }`);

test('empty state, responsive layout, and importer download', async ({page}) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', {level: 1})).toContainText('Bring movement');
  await expect(page.locator('#captureButton')).toBeDisabled();
  await expect(page.locator('#exportButton')).toBeDisabled();
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('link', {name: 'Blender importer'}).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('blender_importer.py');
  expect((await readFile(await download.path(), 'utf8'))).toContain('validate_capture');
  await page.screenshot({path: 'test-results/studio-desktop.png', fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({path: 'test-results/studio-mobile.png', fullPage: true});
  expect(errors).toEqual([]);
});

test('GVHMR setup is explicit about the separate engine, missing setup, and limits', async ({page}) => {
  await page.goto('/');
  await expect(page.locator('label[for="trackingQuality"]')).toContainText('MediaPipe');
  await page.getByRole('link', {name: /GVHMR local backend/}).click();
  await expect(page.getByRole('heading', {level: 1})).toContainText('GVHMR model setup');
  await expect(page.getByText(/Pretrained GVHMR inference on this machine is not verified/)).toBeVisible();
  await expect(page.getByText(/Body and root only/)).toBeVisible();
  await expect(page.getByText(/The studio's Extract motion button runs the selected engine/)).toBeVisible();
  const event = page.waitForEvent('download');
  await page.getByRole('link', {name: /full setup checklist/}).click();
  const download = await event;
  expect((await readFile(await download.path(), 'utf8'))).toContain('SMPLX_NEUTRAL.npz');
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('video decoding, complete capture, scrubbing, and versioned export', async ({page}) => {
  await page.route('**/tracker.worker.mjs', route => route.fulfill({contentType: 'text/javascript', body: fakeWorker}));
  await page.goto('/');
  await page.selectOption('#captureEngine','mediapipe');
  await page.locator('#fileInput').setInputFiles(fixture);
  await expect(page.locator('#statusTitle')).toHaveText('Ready to capture');
  await expect(page.locator('#captureButton')).toBeEnabled();
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Your capture is ready');
  await expect(page.locator('#captureStats')).toContainText('BODY FRAMES');
  await page.locator('#timeline').fill('0.5');
  await expect(page.locator('#currentTime')).toHaveText('00:00.50');
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#exportButton').click();
  const download = await downloadEvent;
  const result = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(result.format).toBe('framecraft-mocap');
  expect(result.frames).toHaveLength(24);
  expect(result.frames[0].body.world).toHaveLength(33);
  expect(result.frames[0].face.landmarks).toHaveLength(478);
  expect(result.frames[0].hands[0].world).toHaveLength(21);
  expect(result.frames.at(-1).time).toBeLessThan(1);
  await page.screenshot({path: 'test-results/studio-captured.png', fullPage: true});
});

test('RTMW3D is default and its guide requires no setup commands', async ({page}) => {
  await page.goto('/');
  await expect(page.locator('#captureEngine')).toHaveValue('rtmw3d');
  await page.getByRole('link', {name:/Temporal body \+ RTMW3D/}).click();
  await expect(page.getByRole('heading', {level:1})).toContainText('Just choose your video');
  await expect(page.getByRole('heading', {name:'No terminal commands. No manual model setup.'})).toBeVisible();
  await expect(page.getByText('Absolute scale and travel are still estimated.')).toBeVisible();
  await expect(page.getByText(/Your selected video is sent only to this PC's loopback server/)).toBeVisible();
  const guide = await page.request.get('/RTMW3D.md');
  expect(guide.status()).toBe(200);
  expect(await guide.text()).toContain('68-point');
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
});

test('cancel terminates loading and does not offer a partial export', async ({page}) => {
  await page.route('**/tracker.worker.mjs', route => route.fulfill({contentType: 'text/javascript', body: 'self.onmessage = () => {};'}));
  await page.goto('/');
  await page.selectOption('#captureEngine','mediapipe');
  await page.locator('#fileInput').setInputFiles(fixture);
  await expect(page.locator('#captureButton')).toBeEnabled();
  await page.locator('#captureButton').click();
  await page.locator('#cancelButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Capture cancelled');
  await expect(page.locator('#exportButton')).toBeDisabled();
  await expect(page.locator('#captureButton')).toBeEnabled();
});

test('invalid video and worker failure recover with useful errors', async ({page}) => {
  await page.goto('/');
  await page.selectOption('#captureEngine','mediapipe');
  await page.locator('#fileInput').setInputFiles({name: 'invalid.mp4', mimeType: 'video/mp4', buffer: Buffer.from('not a video')});
  await expect(page.locator('#statusTitle')).toHaveText('Could not open this video');
  await expect(page.locator('#captureButton')).toBeDisabled();
  await page.route('**/tracker.worker.mjs', route => route.fulfill({contentType: 'text/javascript',
    body: 'self.onmessage = ({data}) => self.postMessage({id: data.id, error: "Model unavailable"});'}));
  await page.locator('#fileInput').setInputFiles(fixture);
  await expect(page.locator('#captureButton')).toBeEnabled();
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Capture could not finish');
  await expect(page.locator('#statusText')).toContainText('Model unavailable');
  await expect(page.locator('#captureButton')).toBeEnabled();
  await expect(page.locator('#exportButton')).toBeDisabled();
});

test('all layers disabled prevents capture', async ({page}) => {
  await page.goto('/');
  await page.selectOption('#captureEngine','mediapipe');
  await page.locator('#fileInput').setInputFiles(fixture);
  await expect(page.locator('#captureButton')).toBeEnabled();
  for (const id of ['captureBody', 'captureHands', 'captureFace']) await page.locator(`#${id}`).uncheck();
  await expect(page.locator('#captureButton')).toBeDisabled();
});

test('ground travel is previewed and exported; disabling it preserves in-place export', async ({page}) => {
  await page.route('**/tracker.worker.mjs', route => route.fulfill({contentType: 'text/javascript', body: fakeGroundWorker}));
  await page.goto('/');
  await page.selectOption('#captureEngine','mediapipe');
  await page.locator('#fileInput').setInputFiles(fixture);
  await expect(page.locator('#captureButton')).toBeEnabled();
  await page.locator('#cameraPitch').fill('0');
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Your capture is ready');
  await expect(page.locator('#rootPreviewPanel')).toBeVisible();
  await expect(page.locator('#rootCoverage')).toContainText('24 / 24');
  await page.locator('#timeline').fill('0.8');
  await expect(page.locator('#rootReadout')).toContainText('Hip height');
  let event = page.waitForEvent('download');
  await page.locator('#exportButton').click();
  let download = await event;
  const result = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(result.rootMotion.enabled).toBe(true);
  expect(result.rootMotion.pitch).toBe(0);
  expect(result.frames.at(-1).root.position.x).toBeGreaterThan(.25);
  expect(result.frames.at(-1).root.position.z).toBeCloseTo(1, 1);
  await page.screenshot({path: 'test-results/studio-ground-motion.png', fullPage: true});
  await page.locator('#captureRoot').uncheck();
  await expect(page.locator('#cameraFov')).toBeDisabled();
  await page.locator('#captureButton').click();
  await expect(page.locator('#statusTitle')).toHaveText('Your capture is ready');
  await expect(page.locator('#rootPreviewPanel')).toBeHidden();
  event = page.waitForEvent('download');
  await page.locator('#exportButton').click();
  download = await event;
  const inPlace = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(inPlace.rootMotion).toBeUndefined();
  expect(inPlace.frames[0].root).toBeUndefined();
  const repairEvent = page.waitForEvent('download');
  await page.locator('#repairCaptureInput').setInputFiles({name: 'legacy.mocap.json',
    mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(inPlace))});
  const repairedDownload = await repairEvent;
  expect(repairedDownload.suggestedFilename()).toBe('legacy.grounded.mocap.json');
  const repaired = JSON.parse(await readFile(await repairedDownload.path(), 'utf8'));
  expect(repaired.rootMotion.enabled).toBe(true);
  expect(repaired.frames.at(-1).root.position.x).toBeGreaterThan(.25);
  await expect(page.locator('#statusTitle')).toHaveText('Repaired capture downloaded');
  await page.locator('#captureBody').uncheck();
  await expect(page.locator('#captureRoot')).toBeDisabled();
});

// Opt-in integration test: downloads actual AI models and runs all three models.
test('real MediaPipe models execute in the worker', async ({page}) => {
  test.skip(!process.env.REAL_MODELS, 'Set REAL_MODELS=1 to run the network-dependent integration test.');
  test.setTimeout(300000);
  const logs = [];
  page.on('console', message => logs.push(message.text()));
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const worker = new Worker('./tracker.worker.mjs');
    const request = data => new Promise((resolve, reject) => {
      worker.onmessage = ({data: message}) => {
        if (message.type === 'status') return;
        if (message.error) reject(new Error(message.error));
        else resolve(message.result);
      };
      worker.onerror = event => reject(new Error(event.message));
      worker.postMessage(data, data.bitmap ? [data.bitmap] : []);
    });
    try {
      const init = await request({id: 1, type: 'init', config: {body: true, hands: true, face: true}});
      const canvas = new OffscreenCanvas(256, 256);
      canvas.getContext('2d').fillRect(0, 0, 256, 256);
      const bitmap = canvas.transferToImageBitmap();
      const frame = await request({id: 2, type: 'frame', bitmap, time: 0});
      const detected = {};
      let poseSize;
      for (const [index, asset] of ['pose.jpg', 'portrait.jpg', 'right_hands.jpg'].entries()) {
        const response = await fetch(`https://storage.googleapis.com/mediapipe-assets/${asset}`);
        if (!response.ok) throw new Error(`Test asset unavailable: ${asset}`);
        const image = await createImageBitmap(await response.blob());
        if (asset === 'pose.jpg') poseSize = {width: image.width, height: image.height};
        detected[asset] = await request({id: index + 3, type: 'frame', bitmap: image, time: index + 1});
      }
      const {reconstructRootMotion} = await import('./root-motion.mjs');
      const grounded = reconstructRootMotion([detected['pose.jpg']], {...poseSize, fov: 60, pitch: 10});
      return {edges: init.faceEdges.length, frame, detected,
        root: {info: grounded.info, sample: grounded.frames[0].root}};
    } finally { worker.terminate(); }
  });
  expect(result.edges).toBeGreaterThan(1000);
  expect(result.frame).toEqual({time: 0, body: null, hands: [], face: null});
  expect(result.detected['pose.jpg'].body.world).toHaveLength(33);
  expect(result.detected['portrait.jpg'].face.landmarks).toHaveLength(478);
  expect(Object.keys(result.detected['portrait.jpg'].face.blendshapes)).toHaveLength(52);
  expect(result.detected['right_hands.jpg'].hands.length).toBeGreaterThan(0);
  expect(result.detected['right_hands.jpg'].hands[0].world).toHaveLength(21);
  expect(result.detected['right_hands.jpg'].hands[0].side).toBe('Right');
  expect(result.root.info.enabled).toBe(true);
  expect(Number.isFinite(result.root.sample.position.z)).toBe(true);
  expect(result.root.sample.status).toBe('tracked');
  console.log('Real models executed; face edges:', result.edges);
});
