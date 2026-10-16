// Models execute off the UI thread. Video frames never leave the browser.
const VERSION = '0.10.21';
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const MODEL_ROOT = 'https://storage.googleapis.com/mediapipe-models';
let pose, hands, face;
let associateHands;

function points(landmarks) {
  return landmarks?.map(point => ({x: point.x, y: point.y, z: point.z,
    ...(Number.isFinite(point.visibility) ? {visibility: point.visibility} : {})})) ?? [];
}

async function createTask(Type, files, options, model, label) {
  self.postMessage({type: 'status', text: `Loading ${label} model… First use needs an internet connection.`});
  const baseOptions = {modelAssetPath: `${MODEL_ROOT}/${model}`};
  try {
    return await Type.createFromOptions(files, {...options, baseOptions: {...baseOptions, delegate: 'GPU'}});
  } catch (gpuError) {
    console.info(`${label}: GPU unavailable, trying CPU`, gpuError);
    return Type.createFromOptions(files, {...options, baseOptions: {...baseOptions, delegate: 'CPU'}});
  }
}

async function initialize(config) {
  ({associateHands} = await import('./core.mjs'));
  const {FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker} = await import(`${CDN}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const common = {runningMode: 'VIDEO', minTrackingConfidence: 0.5};
  const poseVariant = config.quality === 'balanced' ? 'full' : 'heavy';
  if (config.body) pose = await createTask(PoseLandmarker, files,
    {...common, numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5},
    `pose_landmarker/pose_landmarker_${poseVariant}/float16/1/pose_landmarker_${poseVariant}.task`, `${poseVariant} body`);
  if (config.hands) hands = await createTask(HandLandmarker, files,
    {...common, numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5},
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand');
  if (config.face) face = await createTask(FaceLandmarker, files,
    {...common, numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true},
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face');
  const seen = new Set();
  const faceEdges = config.face ? FaceLandmarker.FACE_LANDMARKS_TESSELATION.flatMap(({start, end}) => {
    const key = [start, end].sort((a, b) => a - b).join(':');
    if (seen.has(key)) return [];
    seen.add(key);
    return [[start, end]];
  }) : [];
  return {faceEdges};
}

function track(bitmap, time) {
  const timestamp = time * 1000;
  const result = {time, body: null, hands: [], face: null};
  if (pose) {
    const p = pose.detectForVideo(bitmap, timestamp);
    if (p.landmarks[0]) result.body = {landmarks: points(p.landmarks[0]), world: points(p.worldLandmarks[0])};
  }
  if (hands) {
    const h = hands.detectForVideo(bitmap, timestamp);
    result.hands = h.landmarks.map((landmarks, index) => ({
      side: h.handedness[index][0].categoryName,
      score: h.handedness[index][0].score, landmarks: points(landmarks), world: points(h.worldLandmarks[index])
    }));
    result.hands = associateHands(result.hands, result.body);
  }
  if (face) {
    const f = face.detectForVideo(bitmap, timestamp);
    if (f.faceLandmarks[0]) result.face = {
      landmarks: points(f.faceLandmarks[0]),
      blendshapes: Object.fromEntries((f.faceBlendshapes[0]?.categories ?? []).map(c => [c.categoryName, c.score])),
      matrix: Array.from(f.facialTransformationMatrixes[0]?.data ?? [])
    };
  }
  return result;
}

self.onmessage = async ({data}) => {
  try {
    let result;
    if (data.type === 'init') result = await initialize(data.config);
    else if (data.type === 'frame') {
      try { result = track(data.bitmap, data.time); }
      finally { data.bitmap.close(); }
    } else throw new Error('Unknown tracking request.');
    self.postMessage({id: data.id, result});
  } catch (error) {
    self.postMessage({id: data.id, error: error.message || String(error)});
  }
};
