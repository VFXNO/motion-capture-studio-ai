import {POSE_CONNECTIONS, HAND_CONNECTIONS, formatTime, sampleTimes,
  nearestFrame, captureSummary, createCapture} from './core.mjs';
import {reconstructRootMotion, rebuildCaptureRoot} from './root-motion.mjs';
import {refineCaptureFrames} from './refinement.mjs';
import {localSession, captureLocalVideo, blenderSceneBlob, fitNativeTracking} from './local-capture.mjs';

const $ = id => document.getElementById(id);
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const sourceCanvas = document.createElement('canvas');
const sourceContext = sourceCanvas.getContext('2d', {alpha: false});
const settings = ['captureBody', 'captureHands', 'captureFace', 'sampleRate', 'smoothing',
  'captureRoot', 'cameraFov', 'cameraPitch', 'footStabilization', 'trackingQuality', 'cleanSavedCapture',
  'captureEngine', 'actorHeight', 'computeDevice', 'createBlend'];
let sourceFile = null, sourceURL = null, capture = null, running = false, loading = false;
let activeWorker = null, controller = null, loadController = null;
let frames = [], faceEdges = [], currentFPS = 24;
let capabilities = null, blendJob = null;
const localEngine = () => $('captureEngine').value === 'rtmw3d';

function status(title, text, error = false) {
  $('statusTitle').textContent = title;
  $('statusText').textContent = text;
  $('statusIcon').textContent = error ? '!' : '◇';
  document.querySelector('.status-panel').classList.toggle('error', error);
}

function updateControls() {
  const busy = running || loading;
  const local = localEngine();
  settings.forEach(id => { $(id).disabled = busy; });
  $('browserSettings').hidden = local;
  $('localSettings').hidden = !local;
  $('captureBody').disabled = busy || local;
  $('trackingQuality').disabled = busy || local;
  $('createBlend').disabled = busy || !local || capabilities?.blenderAvailable === false;
  $('faceTitle').textContent = local ? 'Facial landmarks' : 'Face & expression';
  $('faceDetails').textContent = local ? '68 landmarks · no expression scores' : '478 landmarks · 52 expression scores';
  $('engineNote').textContent = local ? (capabilities?.installed
    ? 'Installed on this PC · setup is checked automatically when you capture.'
    : 'First capture installs everything automatically · models download once.')
    : 'Runs in the browser · MediaPipe models download on first use.';
  $('captureRoot').disabled = busy || !$('captureBody').checked;
  for (const id of ['cameraFov', 'cameraPitch', 'footStabilization']) {
    $(id).disabled = busy || !$('captureBody').checked || !$('captureRoot').checked;
  }
  if (local) $('cameraFov').disabled = busy;
  $('captureButton').disabled = busy || !sourceFile || !settings.slice(0, 3).some(id => $(id).checked);
  $('captureButton').hidden = running;
  $('cancelButton').hidden = !running;
  $('replaceButton').disabled = busy;
  $('fileInput').disabled = busy;
  $('repairCaptureButton').disabled = busy;
  $('repairCaptureInput').disabled = busy;
  $('timeline').disabled = !sourceFile || busy;
  $('playButton').disabled = !sourceFile || busy;
  $('exportButton').disabled = busy || !capture;
  $('blendButton').hidden = !blendJob;
  $('blendButton').disabled = busy || !blendJob;
}

class Tracker {
  constructor(onStatus) {
    // MediaPipe's WASM loader uses importScripts, so this must be a classic worker.
    // Dynamic import() inside the worker still loads the ES-module vision bundle.
    this.worker = new Worker('./tracker.worker.mjs');
    this.pending = new Map();
    this.nextID = 0;
    this.worker.onmessage = ({data}) => {
      if (data.type === 'status') { onStatus(data.text); return; }
      const request = this.pending.get(data.id);
      if (!request) return;
      this.pending.delete(data.id);
      clearTimeout(request.timeout);
      if (data.error) request.reject(new Error(data.error));
      else request.resolve(data.result);
    };
    this.worker.onerror = event => {
      event.preventDefault();
      this.close(new Error(event.message || 'The tracking worker could not start. Use current Chrome or Edge.'));
    };
  }
  request(type, data = {}, transfer = []) {
    const id = ++this.nextID;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => this.close(new Error(
        'Tracking timed out. Check your connection for model downloads, or try a shorter video.')), type === 'init' ? 240000 : 60000);
      this.pending.set(id, {resolve, reject, timeout});
      this.worker.postMessage({id, type, ...data}, transfer);
    });
  }
  close(error = new DOMException('Capture cancelled.', 'AbortError')) {
    this.worker.terminate();
    for (const request of this.pending.values()) {
      clearTimeout(request.timeout);
      request.reject(error);
    }
    this.pending.clear();
  }
}

function waitForVideo(event, action, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException('Cancelled.', 'AbortError')); return; }
    const cleanup = () => {
      clearTimeout(timeout);
      video.removeEventListener(event, done);
      video.removeEventListener('error', fail);
      signal?.removeEventListener('abort', abort);
    };
    const done = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error('This video cannot be decoded. Convert it to MP4 / H.264 or WebM and try again.')); };
    const abort = () => { cleanup(); reject(new DOMException('Cancelled.', 'AbortError')); };
    const timeout = setTimeout(() => { cleanup(); reject(new Error('The video decoder did not respond. Try an MP4 / H.264 file.')); }, 20000);
    video.addEventListener(event, done, {once: true});
    video.addEventListener('error', fail, {once: true});
    signal?.addEventListener('abort', abort, {once: true});
    try { action(); } catch (error) { cleanup(); reject(error); }
  });
}

async function seek(time, signal) {
  if (signal?.aborted) throw new DOMException('Cancelled.', 'AbortError');
  if (Math.abs(video.currentTime - time) < 0.000001 && video.readyState >= 2) return;
  await waitForVideo('seeked', () => { video.currentTime = time; }, signal);
}

function positionOverlay() {
  if (!video.videoWidth) return;
  const stage = $('dropzone');
  const scale = Math.min(stage.clientWidth / video.videoWidth, stage.clientHeight / video.videoHeight);
  canvas.style.width = `${video.videoWidth * scale}px`;
  canvas.style.height = `${video.videoHeight * scale}px`;
  canvas.style.left = `${(stage.clientWidth - video.videoWidth * scale) / 2}px`;
  canvas.style.top = `${(stage.clientHeight - video.videoHeight * scale) / 2}px`;
}
new ResizeObserver(positionOverlay).observe($('dropzone'));

function drawPoints(points, edges, color, showJoints = true) {
  if (!points?.length) return;
  const visible = p => p && (p.visibility ?? 1) >= 0.4 && Number.isFinite(p.x) && Number.isFinite(p.y);
  const lineScale = Math.max(1, canvas.width / 1000);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.6 * lineScale;
  ctx.beginPath();
  for (const [a, b] of edges) {
    if (!visible(points[a]) || !visible(points[b])) continue;
    ctx.moveTo(points[a].x * canvas.width, points[a].y * canvas.height);
    ctx.lineTo(points[b].x * canvas.width, points[b].y * canvas.height);
  }
  ctx.stroke();
  if (showJoints) {
    for (const p of points) {
      if (!visible(p)) continue;
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 2.2 * lineScale, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawFrame(frame) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!$('showOverlay').checked || !frame) return;
  ctx.globalAlpha = 0.35;
  if (frame.face) drawPoints(frame.face.landmarks, faceEdges, '#f3bd81', false);
  ctx.globalAlpha = 1;
  if (frame.body) drawPoints(frame.body.landmarks, POSE_CONNECTIONS, '#c3f36b');
  for (const hand of frame.hands) drawPoints(hand.landmarks, HAND_CONNECTIONS, '#b3a5ff');
}

function refreshPreview() {
  $('currentTime').textContent = formatTime(video.currentTime);
  $('timeline').value = String(video.currentTime || 0);
  if (!running) {
    const frame = nearestFrame(frames, video.currentTime, currentFPS);
    drawFrame(frame);
    drawRootPath(frame);
  }
}

function drawRootPath(frame) {
  if (!capture?.rootMotion?.enabled) return;
  const map = $('rootMap');
  const context = map.getContext('2d');
  const points = frames.filter(item => item.root?.status === 'tracked').map(item => item.root.position);
  if (!points.length) return;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(-1, ...xs) - 0.3, maxX = Math.max(1, ...xs) + 0.3;
  const minY = Math.min(-1, ...ys) - 0.3, maxY = Math.max(1, ...ys) + 0.3;
  const scale = Math.min((map.width - 50) / (maxX - minX), (map.height - 35) / (maxY - minY));
  const project = p => [map.width / 2 + (p.x - (minX + maxX) / 2) * scale,
    map.height / 2 - (p.y - (minY + maxY) / 2) * scale];
  context.clearRect(0, 0, map.width, map.height);
  context.strokeStyle = '#292f39';
  context.lineWidth = 1;
  const grid = Math.max(1, Math.ceil(Math.max(maxX - minX, maxY - minY) / 15));
  context.beginPath();
  for (let x = Math.ceil(minX / grid) * grid; x <= maxX; x += grid) {
    const [screenX] = project({x, y: 0});
    context.moveTo(screenX, 10); context.lineTo(screenX, map.height - 10);
  }
  for (let y = Math.ceil(minY / grid) * grid; y <= maxY; y += grid) {
    const [, screenY] = project({x: 0, y});
    context.moveTo(10, screenY); context.lineTo(map.width - 10, screenY);
  }
  context.stroke();
  context.strokeStyle = '#c3f36b'; context.lineWidth = 2; context.beginPath();
  let connected = false;
  for (const sample of frames) {
    if (sample.root?.status !== 'tracked') { connected = false; continue; }
    const [x, y] = project(sample.root.position);
    if (connected) context.lineTo(x, y); else context.moveTo(x, y);
    connected = true;
  }
  context.stroke();
  const origin = project({x: 0, y: 0});
  context.strokeStyle = '#808999'; context.strokeRect(origin[0] - 3, origin[1] - 3, 6, 6);
  context.fillStyle = '#969ba9'; context.font = '10px monospace';
  context.fillText(`grid ${grid}m · ↑ away`, 12, map.height - 12);
  if (frame?.root) {
    const {position, status: trackStatus, contacts} = frame.root;
    const [x, y] = project(position);
    context.fillStyle = trackStatus === 'held' ? '#f3bd81' : '#c3f36b';
    context.beginPath(); context.arc(x, y, 5, 0, Math.PI * 2); context.fill();
    $('rootReadout').textContent = `X ${position.x.toFixed(2)}m · Y ${position.y.toFixed(2)}m\nHip height ${position.z.toFixed(2)}m\n${trackStatus === 'held' ? 'Tracking lost — root held' : `Foot contact: ${['Left', 'Right'].filter(side => contacts[side]).join(' + ') || 'none'}`}`;
  } else $('rootReadout').textContent = 'No root estimate at this time.';
}

async function loadFile(file) {
  if (!file || running || loading) return;
  video.pause();
  loadController?.abort();
  loadController = new AbortController();
  loading = true;
  sourceFile = null;
  capture = null;
  blendJob = null;
  frames = [];
  faceEdges = [];
  $('captureStats').hidden = true;
  $('progressArea').hidden = true;
  $('rootPreviewPanel').hidden = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  updateControls();
  if (sourceURL) URL.revokeObjectURL(sourceURL);
  sourceURL = URL.createObjectURL(file);
  status('Opening your video', 'Checking the video format and duration…');
  try {
    await waitForVideo('loadeddata', () => { video.src = sourceURL; video.load(); }, loadController.signal);
    if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth) {
      throw new Error('This video has no usable duration. Convert it to a standard MP4 file.');
    }
    sourceFile = file;
    video.hidden = false;
    canvas.hidden = false;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const scale = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight));
    sourceCanvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    sourceCanvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    $('emptyState').hidden = true;
    $('viewerBadge').hidden = false;
    $('dropzone').classList.add('loaded');
    $('dropzone').removeAttribute('role');
    $('dropzone').setAttribute('aria-label', 'Video preview. Use the controls below to play or replace the video.');
    $('videoInfo').textContent = `${file.name} · ${video.videoWidth} × ${video.videoHeight}`;
    $('timeline').max = String(video.duration);
    $('duration').textContent = formatTime(video.duration);
    positionOverlay();
    refreshPreview();
    status('Ready to capture', `${file.name} · ${formatTime(video.duration)}. Choose your layers, then extract motion.`);
  } catch (error) {
    if (error.name !== 'AbortError') status('Could not open this video', error.message, true);
    video.hidden = true;
    canvas.hidden = true;
    $('emptyState').hidden = false;
    $('viewerBadge').hidden = true;
    $('dropzone').classList.remove('loaded');
    $('dropzone').setAttribute('role', 'button');
    $('dropzone').setAttribute('aria-label', 'Choose or drop a recorded video');
    $('videoInfo').textContent = 'NO VIDEO LOADED';
    $('duration').textContent = '00:00.00';
    $('currentTime').textContent = '00:00.00';
  } finally {
    loading = false;
    updateControls();
  }
}

async function extractMotion() {
  if (!sourceFile || running) return;
  const config = {body: $('captureBody').checked, hands: $('captureHands').checked, face: $('captureFace').checked,
    quality: $('trackingQuality').value};
  const local = localEngine();
  if (![config.body, config.hands, config.face].some(Boolean)) return;
  const fps = Number($('sampleRate').value);
  // Limit memory use: full facial capture has hundreds of points on every frame.
  const count = Math.ceil(video.duration * fps);
  if (count > 1800 || (local && video.duration > 60)) {
    status('This clip is too long for one capture', `${count.toLocaleString()} samples requested; the limit is 1,800. Choose a lower frame rate or trim the video first.`, true);
    return;
  }
  const times = sampleTimes(video.duration, fps);
  const smoothing = Number($('smoothing').value);
  const rootEnabled = config.body && $('captureRoot').checked;
  if ((rootEnabled || local) && !['cameraFov', 'cameraPitch'].every(id => $(id).reportValidity())) return;
  if (local && !$('actorHeight').reportValidity()) return;
  const rootOptions = {width: video.videoWidth, height: video.videoHeight,
    fov: Number($('cameraFov').value), pitch: Number($('cameraPitch').value), stabilize: $('footStabilization').checked};
  const originalTime = video.currentTime;
  video.pause();
  running = true;
  controller = new AbortController();
  const {signal} = controller;
  capture = null;
  blendJob = null;
  frames = [];
  currentFPS = fps;
  $('captureStats').hidden = true;
  $('rootPreviewPanel').hidden = true;
  $('progressArea').hidden = false;
  $('progress').value = 0;
  $('progressText').textContent = '0%';
  updateControls();
  status('Preparing the tracking engine', 'Downloading AI models if needed. Your video stays on this device.');
  try {
    let localResult = null;
    if (local) {
      localResult = await captureLocalVideo({file: sourceFile, signal,
        options: {filename: sourceFile.name, duration: video.duration, fps, height: Number($('actorHeight').value),
          fov: rootOptions.fov, device: $('computeDevice').value, makeBlend: $('createBlend').checked},
        onProgress: state => {
          status('Capturing with local temporal AI', state.message);
          $('progress').value = state.progress ?? 0;
          $('progressText').textContent = `${Math.round((state.progress ?? 0) * 100)}%`;
        },
        buildCapture: async native => {
          const result = await fitNativeTracking(native, {pitch: rootOptions.pitch, cleanupStrength: smoothing,
            stabilize: rootOptions.stabilize, rootEnabled}, signal);
          result.metadata.config = {...config, engine: 'rtmw3d-x'};
          result.frames = result.frames.map(frame => ({...frame, hands: config.hands ? frame.hands : [], face: config.face ? frame.face : null}));
          return result;
        }});
      capture = localResult.capture;
      frames = capture.frames;
      faceEdges = capture.faceEdges;
      if (localResult.job.blendReady) blendJob = localResult.job.id;
    } else {
    activeWorker = new Tracker(text => status('Preparing the tracking engine', text));
    const init = await activeWorker.request('init', {config});
    faceEdges = init.faceEdges;
    const started = performance.now();
    for (let index = 0; index < times.length; index++) {
      if (signal.aborted) throw new DOMException('Cancelled.', 'AbortError');
      const time = times[index];
      await seek(time, signal);
      sourceContext.drawImage(video, 0, 0, sourceCanvas.width, sourceCanvas.height);
      const bitmap = await createImageBitmap(sourceCanvas);
      if (signal.aborted) { bitmap.close(); throw new DOMException('Cancelled.', 'AbortError'); }
      const raw = await activeWorker.request('frame', {bitmap, time}, [bitmap]);
      const frame = raw;
      frames.push(frame);
      drawFrame(frame);
      const progress = (index + 1) / times.length;
      $('progress').value = progress;
      $('progressText').textContent = `${Math.round(progress * 100)}%`;
      if (index % 8 === 0 || index === times.length - 1) {
        const elapsed = (performance.now() - started) / 1000;
        const remaining = elapsed / (index + 1) * (times.length - index - 1);
        status('Extracting your performance', `Frame ${index + 1} of ${times.length} · about ${Math.ceil(remaining)}s remaining. Preview follows the current sample.`);
      }
    }
    const summary = captureSummary(frames);
    const found = summary.body || summary.hands || summary.face;
    if (!found) throw new Error('No person was detected. Try brighter footage with one visible performer.');
    const refinement = refineCaptureFrames(frames, {strength: smoothing});
    frames = refinement.frames;
    let rootMotion = null;
    if (rootEnabled) {
      const reconstruction = reconstructRootMotion(frames, rootOptions);
      frames = reconstruction.frames;
      rootMotion = reconstruction.info;
    }
    capture = createCapture({filename: sourceFile.name, width: video.videoWidth, height: video.videoHeight,
      duration: video.duration, fps, smoothing, config, refinement: refinement.stats, createdAt: new Date().toISOString(),
      notes: rootMotion?.enabled ? 'Single-person estimates with approximate fixed-camera, flat-ground root travel. Not camera-motion compensated.'
         : 'Single-person, in-place estimates. Missing detections are null.'}, frames, faceEdges, rootMotion);
    }
    const summary = captureSummary(frames);
    const rootMotion = capture.rootMotion;
    $('rootPreviewPanel').hidden = !rootMotion?.enabled;
    if (rootMotion?.enabled) $('rootCoverage').textContent = `${rootMotion.validFrames} / ${frames.length} ROOT SAMPLES`;
    $('captureStats').replaceChildren();
    for (const [value, label] of [[summary.frames, 'SAMPLES'], [summary.body, 'BODY FRAMES'], [summary.hands, 'HAND FRAMES'], [summary.face, 'FACE FRAMES']]) {
      const item = document.createElement('div');
      const strong = document.createElement('strong');
      const small = document.createElement('small');
      strong.textContent = value.toLocaleString();
      small.textContent = label;
      item.append(strong, small);
      $('captureStats').append(item);
    }
    $('captureStats').hidden = false;
    const absent = ['body', 'hands', 'face'].filter(key => config[key] && !summary[key]);
    const resultText = absent.length
      ? `No ${absent.join(' or ')} detected. Other layers are ready to export; scrub the timeline to review them.`
      : `${summary.frames} samples captured. Scrub to review, then download your capture for Blender.`;
    status('Your capture is ready', `${resultText}${rootMotion && !rootMotion.enabled ? ` Root unavailable: ${rootMotion.reason}` : rootMotion?.enabled ? ' Estimated ground travel included.' : ''}${blendJob ? ' Your Blender scene is ready to download—no import script needed.' : ''}${localResult?.job.warning ? ` ${localResult.job.warning}` : ''}`, Boolean(localResult?.job.warning));
  } catch (error) {
    // Never offer an incomplete capture as if the entire video had succeeded.
    capture = null;
    frames = [];
    if (error.name === 'AbortError') status('Capture cancelled', 'No partial export was saved. Your video is ready for another run.');
    else status('Capture could not finish', `${error.message}${local ? ' No MediaPipe fallback was used.' : ' Model downloads require access to jsDelivr and Google Storage.'}`, true);
  } finally {
    activeWorker?.close();
    activeWorker = null;
    running = false;
    controller = null;
    $('progressArea').hidden = true;
    updateControls();
    try { await seek(originalTime); } catch { /* User can replace a failed decoder. */ }
    refreshPreview();
  }
}

function chooseFile() { if (!running && !loading) $('fileInput').click(); }
$('dropzone').addEventListener('click', () => { if (!sourceFile) chooseFile(); });
$('dropzone').addEventListener('keydown', event => {
  if (!sourceFile && ['Enter', ' '].includes(event.key)) { event.preventDefault(); chooseFile(); }
});
$('replaceButton').addEventListener('click', chooseFile);
$('fileInput').addEventListener('change', event => {
  const file = event.target.files[0];
  event.target.value = '';
  loadFile(file);
});
for (const eventName of ['dragenter', 'dragover']) $('dropzone').addEventListener(eventName, event => {
  event.preventDefault();
  if (!running && !loading) $('dropzone').classList.add('dragging');
});
for (const eventName of ['dragleave', 'drop']) $('dropzone').addEventListener(eventName, event => {
  event.preventDefault();
  $('dropzone').classList.remove('dragging');
  if (eventName === 'drop') loadFile(event.dataTransfer.files[0]);
});
// Prevent dropping outside the target from navigating away and losing a capture.
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
$('playButton').addEventListener('click', async () => {
  if (video.paused) {
    try { await video.play(); } catch (error) { status('Playback unavailable', error.message, true); }
  } else video.pause();
});
video.addEventListener('play', () => { $('playButton').textContent = 'Ⅱ'; $('playButton').setAttribute('aria-label', 'Pause video'); });
video.addEventListener('pause', () => { $('playButton').textContent = '▶'; $('playButton').setAttribute('aria-label', 'Play video'); });
video.addEventListener('timeupdate', refreshPreview);
video.addEventListener('seeked', refreshPreview);
if ('requestVideoFrameCallback' in video) {
  const render = () => { if (!running) refreshPreview(); video.requestVideoFrameCallback(render); };
  video.requestVideoFrameCallback(render);
}
$('timeline').addEventListener('input', event => { video.pause(); video.currentTime = Number(event.target.value); });
$('showOverlay').addEventListener('change', () => drawFrame(nearestFrame(frames, video.currentTime, currentFPS)));
$('smoothing').addEventListener('input', () => { $('smoothingValue').textContent = `${Math.round(Number($('smoothing').value) * 100)}%`; });
settings.slice(0, 3).forEach(id => $(id).addEventListener('change', updateControls));
$('captureRoot').addEventListener('change', updateControls);
$('captureEngine').addEventListener('change', () => {
  if (localEngine()) $('captureBody').checked = true;
  updateControls();
});
$('blendButton').addEventListener('click', async () => {
  if (!blendJob) return;
  $('blendButton').disabled = true;
  try {
    const blob = await blenderSceneBlob(blendJob);
    downloadBlob(blob, `${sourceFile.name.replace(/\.[^.]+$/, '')}.blend`);
  } catch (error) { status('Could not download Blender scene', error.message, true); }
  finally { updateControls(); }
});
$('repairCaptureButton').addEventListener('click', () => $('repairCaptureInput').click());
$('repairCaptureInput').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file || running || loading) return;
  try {
    if (file.size > 512 * 1024 * 1024) throw new Error('The capture exceeds the 512 MB limit.');
    if (!['cameraFov', 'cameraPitch'].every(id => $(id).reportValidity())) return;
    loading = true;
    updateControls();
    status('Rebuilding saved root motion', 'Reading your capture locally. No video or AI-model download is needed.');
    const saved = JSON.parse(await file.text());
    const repaired = rebuildCaptureRoot(saved, {fov: Number($('cameraFov').value),
      pitch: Number($('cameraPitch').value), stabilize: $('footStabilization').checked,
      cleanupStrength: $('cleanSavedCapture').checked ? Number($('smoothing').value) : 0});
    const name = file.name.replace(/(?:\.mocap)?\.json$/i, '') + '.grounded.mocap.json';
    downloadCapture(repaired, name);
    status('Repaired capture downloaded', `Import ${name} in Blender—not the old JSON. The original file is unchanged. Use the latest Blender importer.`);
  } catch (error) { status('Could not rebuild root motion', error.message, true); }
  finally { loading = false; updateControls(); }
});
$('captureButton').addEventListener('click', extractMotion);
$('cancelButton').addEventListener('click', () => { controller?.abort(); activeWorker?.close(); });
$('exportButton').addEventListener('click', () => {
  if (!capture) return;
  try {
    downloadCapture(capture, `${sourceFile.name.replace(/\.[^.]+$/, '')}.mocap.json`);
  } catch (error) { status('Export failed', `The capture may be too large for this browser. ${error.message}`, true); }
});

function downloadCapture(data, filename) {
  const json = JSON.stringify(data, (_key, value) => typeof value === 'number' ? Math.round(value * 1e6) / 1e6 : value);
  downloadBlob(new Blob([json], {type: 'application/json'}), filename);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
window.addEventListener('beforeunload', () => {
  controller?.abort();
  activeWorker?.close();
  if (sourceURL) URL.revokeObjectURL(sourceURL);
});
localSession().then(session => {
  capabilities = session;
  $('engineNote').textContent = session.installed ? 'Installed on this PC · setup is checked automatically when you capture.' : 'First capture installs everything automatically · models download once.';
  if (!session.blenderAvailable) {
    $('createBlend').checked = false;
    $('blenderNote').textContent = 'Blender was not detected. Capture JSON is available; install Blender for automatic .blend creation.';
  }
  updateControls();
}).catch(() => {
  $('engineNote').textContent = 'Launch Studio.bat starts the local AI service. No setup commands are needed.';
});
updateControls();
