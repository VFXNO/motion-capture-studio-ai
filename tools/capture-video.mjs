// Optional development CLI. Uses the same local browser pipeline as the app.
import {chromium} from '@playwright/test';
import {readFile, writeFile, stat} from 'node:fs/promises';
import {resolve} from 'node:path';

const [input, output, fps = '24', baseURL = 'http://127.0.0.1:8765'] = process.argv.slice(2);
let browser;
try {
  if (!input || !output || !['12', '24', '30'].includes(fps)) throw new Error('Usage: node tools/capture-video.mjs INPUT_VIDEO NEW_OUTPUT.mocap.json [FPS=24] [LOCAL_APP_URL]');
  if (await stat(output).then(() => true, () => false)) throw new Error('Output already exists; choose a new filename.');
  browser = await chromium.launch({headless: true});
  const page = await browser.newPage({acceptDownloads: true});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(baseURL);
  await page.locator('#fileInput').setInputFiles(resolve(input));
  await page.locator('#captureButton').waitFor({state: 'visible'});
  await page.waitForFunction(() => !document.getElementById('captureButton').disabled, undefined, {timeout: 30000});
  await page.locator('#trackingQuality').selectOption('accurate');
  await page.locator('#sampleRate').selectOption(fps);
  await page.locator('#captureButton').click();
  await page.waitForFunction(() => ['Your capture is ready', 'Capture could not finish', 'This clip is too long for one capture'].includes(
    document.getElementById('statusTitle').textContent), undefined, {timeout: 600000});
  if (await page.locator('#exportButton').isDisabled()) throw new Error(await page.locator('#statusText').textContent());
  const event = page.waitForEvent('download');
  await page.locator('#exportButton').click();
  const download = await event;
  const bytes = await readFile(await download.path());
  await writeFile(output, bytes, {flag: 'wx'});
  const capture = JSON.parse(bytes.toString('utf8'));
  console.log(JSON.stringify({output: resolve(output), frames: capture.frames.length,
    quality: capture.metadata.config.quality, rootFrames: capture.rootMotion?.validFrames,
    body: capture.frames.filter(f => f.body).length, face: capture.frames.filter(f => f.face).length,
    hands: capture.frames.filter(f => f.hands.length).length, cleanup: capture.metadata.refinement,
    browserErrors: errors}, null, 2));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
} finally { await browser?.close(); }
