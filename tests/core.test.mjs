import test from 'node:test';
import assert from 'node:assert/strict';
import {formatTime, sampleTimes, smoothFrame, nearestFrame, captureSummary, createCapture, associateHands} from '../core.mjs';

const point = (x, visibility = 1) => ({x, y: x, z: x, visibility});
const frame = (time, x) => ({time,
  body: {landmarks: [point(x)], world: [point(x)]},
  hands: [{side: 'Left', landmarks: [point(x)], world: [point(x)]}],
  face: {landmarks: [point(x)], blendshapes: {eyeBlinkLeft: x}, matrix: []}});

test('time formatting carries centiseconds without rounding to 60 seconds', () => {
  assert.equal(formatTime(59.999), '00:59.99');
  assert.equal(formatTime(60), '01:00.00');
  assert.equal(formatTime(1.15), '00:01.15');
  assert.equal(formatTime(Infinity), '00:00.00');
});
test('sampling covers a clip without sampling its undecodable end', () => {
  assert.deepEqual(sampleTimes(1, 4), [0, 0.25, 0.5, 0.75]);
  assert.deepEqual(sampleTimes(0.01, 24), [0]);
  assert.equal(sampleTimes(1.01, 4).at(-1), 1);
  for (const value of [NaN, Infinity, 0, -1]) assert.throws(() => sampleTimes(value, 24));
  assert.throws(() => sampleTimes(1, 0));
});
test('smoothing filters body, matching hands, face, and expression scores without mutation', () => {
  const previous = frame(0, 0);
  const current = frame(1, 1);
  const result = smoothFrame(previous, current, 0.4);
  assert.equal(result.body.world[0].x, 0.6);
  assert.equal(result.hands[0].world[0].x, 0.6);
  assert.equal(result.face.blendshapes.eyeBlinkLeft, 0.6);
  assert.equal(current.body.world[0].x, 1);
  assert.equal(previous.face.blendshapes.eyeBlinkLeft, 0);
});
test('missing tracks stay missing, and reappearing tracks do not blend across a gap', () => {
  const gap = {time: 1, body: null, face: null, hands: []};
  assert.deepEqual(smoothFrame(frame(0, 0), gap, 0.5), gap);
  assert.deepEqual(smoothFrame(gap, frame(2, 1), 0.5), frame(2, 1));
});
test('opposite hand labels are not blended and low-confidence joints are not dragged', () => {
  const current = frame(1, 1), old = frame(0, 0);
  current.hands[0].side = 'Right';
  old.body.world[0].visibility = 0.1;
  const result = smoothFrame(old, current, 0.8);
  assert.equal(result.hands[0].world[0].x, 1);
  assert.equal(result.body.world[0].x, 1);
});
test('timeline lookup never shows stale last-frame tracking far past a partial capture', () => {
  const frames = [frame(0, 0), frame(0.5, 1)];
  assert.equal(nearestFrame(frames, 0.4, 2), frames[1]);
  assert.equal(nearestFrame(frames, 3, 2), null);
  assert.equal(nearestFrame([], 0, 2), null);
});
test('export includes versioned schema and detection coverage', () => {
  const frames = [frame(0, 0), {time: 0.5, body: null, hands: [], face: null}];
  assert.deepEqual(captureSummary(frames), {frames: 2, body: 1, hands: 1, face: 1});
  const result = createCapture({fps: 2}, frames, [[0, 1]]);
  assert.equal(result.format, 'framecraft-mocap');
  assert.equal(result.version, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(result)).faceEdges, [[0, 1]]);
});

test('hand association uses body wrists, including crossed hands and duplicate model labels', () => {
  const landmarks = Array.from({length: 33}, () => point(0, 0));
  landmarks[15] = {x: 0.8, y: 0.5, visibility: 1};
  landmarks[16] = {x: 0.2, y: 0.5, visibility: 1};
  const hands = [{side: 'Right', score: 0.9, landmarks: [{x: 0.79, y: 0.5}]},
    {side: 'Right', score: 0.8, landmarks: [{x: 0.21, y: 0.5}]}];
  assert.deepEqual(associateHands(hands, {landmarks}).map(h => h.side), ['Left', 'Right']);
  assert.equal(hands[0].side, 'Right');
  assert.equal(associateHands(hands, null).length, 1);
  assert.equal(associateHands(hands, null)[0].score, 0.9);
});
