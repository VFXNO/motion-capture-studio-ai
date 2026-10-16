import test from 'node:test';
import assert from 'node:assert/strict';
import {refineCaptureFrames} from '../refinement.mjs';

const p = x => ({x, y: 0, z: 0, visibility: 1});
const frame = (i, x) => ({time: i / 30, body: {world: Array.from({length: 33}, () => p(x)),
  landmarks: Array.from({length: 33}, () => p(x))}, hands: [], face: null});
const rms = values => Math.sqrt(values.reduce((sum, n) => sum + n * n, 0) / values.length);

test('offline cleanup reduces synthetic high-frequency noise without delaying a slow motion peak', () => {
  const truth = Array.from({length: 121}, (_, i) => .1 * Math.cos((i - 60) * Math.PI / 120));
  const source = truth.map((value, i) => frame(i, value + .015 * Math.sin(i * Math.PI * .7)));
  const snapshot = JSON.stringify(source);
  const result = refineCaptureFrames(source).frames;
  const errors = result.map((f, i) => f.body.world[0].x - truth[i]);
  assert.ok(rms(errors.slice(10, -10)) < rms(source.slice(10, -10).map((f, i) => f.body.world[0].x - truth[i + 10])) * .65);
  const peak = result.reduce((best, f, i) => f.body.world[0].x > result[best].body.world[0].x ? i : best, 0);
  assert.ok(Math.abs(peak - 60) <= 4);
  assert.equal(JSON.stringify(source), snapshot);
});

test('isolated large spikes are corrected but sustained fast movement remains', () => {
  const source = Array.from({length: 31}, (_, i) => frame(i, i === 8 ? 1 : i >= 16 && i <= 22 ? .5 : 0));
  const {frames, stats} = refineCaptureFrames(source);
  assert.ok(frames[8].body.world[0].x < .05);
  assert.ok(frames[19].body.world[0].x > .47);
  assert.ok(stats.outliersCorrected > 0);
});

test('missing tracks and low-confidence joints are never interpolated across', () => {
  const source = Array.from({length: 20}, (_, i) => frame(i, i < 10 ? 0 : 1));
  source[9].body = null;
  source[10].body.world[0].visibility = .1;
  const {frames} = refineCaptureFrames(source);
  assert.equal(frames[9].body, null);
  assert.equal(frames[8].body.world[0].x, 0);
  assert.equal(frames[11].body.world[0].x, 1);
  assert.equal(frames[10].body.world[0].visibility, .1);
});

test('cleanup can be disabled and invalid strengths are rejected', () => {
  const source = [frame(0, 1)];
  assert.equal(refineCaptureFrames(source, {strength: 0}).frames, source);
  assert.throws(() => refineCaptureFrames(source, {strength: NaN}));
  assert.throws(() => refineCaptureFrames(source, {strength: 1}));
});
