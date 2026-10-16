import test from 'node:test';
import assert from 'node:assert/strict';
import {cameraToGround, estimateCameraTranslation, reconstructRootMotion, rebuildCaptureRoot} from '../root-motion.mjs';

const options = {width: 1280, height: 720, fov: 60, pitch: 10};
const near = (actual, expected, tolerance = 1e-6) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

function toCamera(point, pitch = options.pitch) {
  const angle = pitch * Math.PI / 180;
  return {x: point.x, y: -Math.sin(angle) * point.y - Math.cos(angle) * point.z,
    z: Math.cos(angle) * point.y - Math.sin(angle) * point.z};
}

function syntheticFrame(index, {x = 0, y = 0, height = 1, footShift = 0, hideFeet = false, pitch = options.pitch} = {}) {
  const local = Array.from({length: 33}, () => ({x: 0, y: 0, z: 0.65}));
  for (const [side, indices] of [[-1, [11, 13, 23, 25, 27, 29, 31]], [1, [12, 14, 24, 26, 28, 30, 32]]]) {
    const values = [[side * .22, 0, .5], [side * .4, .02, .2], [side * .11, 0, 0],
      [side * .11, .03, -.5], [side * .11 - footShift, 0, -.9],
      [side * .11 - footShift, -.04, -1], [side * .11 - footShift, .16, -1]];
    indices.forEach((joint, i) => { local[joint] = Object.fromEntries(['x', 'y', 'z'].map((axis, a) => [axis, values[i][a]])); });
  }
  const world = local.map(point => toCamera(point, pitch));
  const translation = toCamera({x, y: y + 5, z: height - 2}, pitch);
  const focal = options.width / (2 * Math.tan(options.fov * Math.PI / 360));
  const landmarks = world.map((point, i) => {
    const depth = point.z + translation.z;
    return {x: .5 + (point.x + translation.x) / depth * focal / options.width,
      y: .5 + (point.y + translation.y) / depth * focal / options.height, z: 0,
      visibility: hideFeet && i >= 27 ? 0.1 : 1};
  });
  return {time: index / 24, body: {world, landmarks}, hands: [], face: null};
}

test('camera pitch transform is Z-up and has the expected sign', () => {
  const p = {x: .5, y: 4, z: -1};
  const restored = cameraToGround(toCamera(p), 10);
  for (const axis of ['x', 'y', 'z']) near(restored[axis], p[axis]);
  assert.deepEqual(cameraToGround({x: 1, y: 2, z: 3}, 0), {x: 1, y: 3, z: -2});
});

test('perspective solver recovers lateral travel and depth, not just image hip displacement', () => {
  for (const [x, y, height] of [[0, 0, 1], [1.2, 2, 1], [-.7, -.5, 1.4]]) {
    const solved = estimateCameraTranslation(syntheticFrame(0, {x, y, height}).body, options);
    assert.ok(solved);
    const ground = cameraToGround(solved.position, options.pitch);
    near(ground.x, x); near(ground.y, y + 5); near(ground.z, height - 2);
    assert.ok(solved.confidence > .99);
  }
});

test('robust root fit resists a single incorrectly detected elbow', () => {
  const frame = syntheticFrame(0, {x: .2});
  frame.body.landmarks[13].x += .25;
  frame.body.landmarks[13].y -= .15;
  const solved = estimateCameraTranslation(frame.body, options);
  assert.ok(solved);
  const ground = cameraToGround(solved.position, options.pitch);
  near(ground.x, .2, .04);
  near(ground.y, 5, .12);
});

test('root path translates across the floor with feet at Z=0 and without vertical duplication', () => {
  const frames = Array.from({length: 48}, (_, i) => syntheticFrame(i, {x: i * .02, y: i * .01}));
  const result = reconstructRootMotion(frames, {...options, stabilize: true});
  assert.equal(result.info.enabled, true);
  assert.equal(result.info.validFrames, 48);
  near(result.frames[0].root.position.x, 0);
  near(result.frames[0].root.position.z, 1);
  assert.ok(result.frames.at(-1).root.position.x > .75);
  assert.ok(result.frames.at(-1).root.position.y > .35);
  for (const frame of result.frames) near(frame.root.position.z - 1, 0, .015);
  assert.equal(frames[0].root, undefined, 'input is not mutated');
});

test('planted foot compensation reduces sliding as hips move over a stationary sole', () => {
  const frames = Array.from({length: 24}, (_, i) => syntheticFrame(i, {x: i * .008, footShift: i * .008}));
  const {frames: result} = reconstructRootMotion(frames, options);
  const last = result.at(-1);
  const foot = last.root.position.x - 23 * .008;
  near(foot, 0, .015);
  assert.equal(last.root.contacts.Left, true);
});

test('jumps keep their height instead of snapping every frame to the floor', () => {
  const frames = Array.from({length: 60}, (_, i) => syntheticFrame(i, {height: i >= 18 && i < 42 ? 1.5 : 1}));
  const {frames: result} = reconstructRootMotion(frames, options);
  assert.ok(result[35].root.position.z > 1.45);
  assert.equal(result[35].root.contacts.Left, false);
});

test('lost body detections hold the root and clear contacts without bridging a foot lock', () => {
  const frames = [syntheticFrame(0), {time: 1 / 24, body: null, hands: [], face: null}, syntheticFrame(2, {x: .2})];
  const {frames: result} = reconstructRootMotion(frames, options);
  assert.equal(result[1].root.status, 'held');
  assert.equal(result[1].root.confidence, 0);
  assert.deepEqual(result[1].root.position, result[0].root.position);
  near(result[2].root.position.x, .2);
});

test('missing feet or degenerate geometry fails explicitly rather than fabricating a root', () => {
  const frame = syntheticFrame(0, {hideFeet: true});
  const result = reconstructRootMotion([frame], options);
  assert.equal(result.info.enabled, false);
  assert.match(result.info.reason, /feet/);
  const degenerate = syntheticFrame(0);
  degenerate.body.world = degenerate.body.world.map(() => ({x: 0, y: 0, z: 0}));
  assert.equal(estimateCameraTranslation(degenerate.body, options), null);
  assert.throws(() => reconstructRootMotion([frame], {...options, fov: 0}));
});

test('stabilization can be disabled to preserve the perspective estimate', () => {
  const frames = Array.from({length: 12}, (_, i) => syntheticFrame(i, {x: .01 * i}));
  const {frames: result, info} = reconstructRootMotion(frames, {...options, stabilize: false});
  assert.equal(info.stabilize, false);
  assert.ok(result.at(-1).root.position.x > .08);
});

test('older in-place captures can be repaired without altering their source landmarks', () => {
  const original = {format: 'framecraft-mocap', version: 1,
    metadata: {width: options.width, height: options.height, fps: 24},
    faceEdges: [], frames: Array.from({length: 24}, (_, i) => syntheticFrame(i, {x: i * .02}))};
  const snapshot = JSON.stringify(original);
  const repaired = rebuildCaptureRoot(original, options);
  assert.equal(repaired.rootMotion.enabled, true);
  assert.ok(repaired.frames.at(-1).root.position.x > .3);
  assert.deepEqual(repaired.frames[0].body, original.frames[0].body);
  assert.equal(JSON.stringify(original), snapshot);
  assert.match(repaired.metadata.notes, /rebuilt/i);
  assert.throws(() => rebuildCaptureRoot({frames: []}, options));
});
