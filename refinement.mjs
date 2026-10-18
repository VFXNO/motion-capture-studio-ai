// Offline, bidirectional filtering: reduce high-frequency noise without adding
// the frame delay of the old one-way exponential filter. Never bridge gaps.
const axes = ['x', 'y', 'z'];
const distance = (a, b) => Math.hypot(...axes.map(axis => a[axis] - b[axis]));
const interpolate = (a, b, t) => Object.fromEntries(axes.map(axis => [axis, a[axis] + t * (b[axis] - a[axis])]));
const finite = p => p && axes.every(axis => Number.isFinite(p[axis]));

function filterSegment(values, times, strength, spikeLimit, stats) {
  const clean = values.map(point => ({...point}));
  for (let i = 1; i < values.length - 1; i++) {
    const before = values[i - 1], after = values[i + 1];
    const midpoint = interpolate(before, after, (times[i] - times[i - 1]) / (times[i + 1] - times[i - 1]));
    const deviation = distance(values[i], midpoint);
    if (deviation > spikeLimit && distance(before, after) < deviation * .5 &&
      distance(values[i], before) > spikeLimit && distance(values[i], after) > spikeLimit) {
      clean[i] = {...values[i], ...midpoint};
      stats.outliersCorrected++;
    }
  }
  function pass(input, reverse) {
    const result = input.map(point => ({...point}));
    const indices = Array.from({length: input.length}, (_, i) => reverse ? input.length - 1 - i : i);
    let last = indices[0];
    for (const index of indices.slice(1)) {
      const dt = Math.abs(times[index] - times[last]);
      const speed = distance(clean[index], clean[last]) / dt;
      // Open the cutoff during quick movement instead of flattening dance steps.
      const cutoff = 4.5 * (1 - strength) + .7 + 5 * (1 - strength) * speed;
      const alpha = 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
      result[index] = {...input[index], ...interpolate(result[last], input[index], alpha)};
      last = index;
    }
    return result;
  }
  return pass(pass(clean, false), true);
}

export function refineCaptureFrames(frames, {strength = .35} = {}) {
  if (!Number.isFinite(strength) || strength < 0 || strength > .85) throw new Error('Refinement strength must be 0–0.85.');
  const stats = {method: 'bidirectional-adaptive-v1', strength, outliersCorrected: 0};
  if (!strength || !frames.length) return {frames, stats};
  // Clone mutable point arrays while retaining other metadata and missing tracks.
  const output = structuredClone(frames);
  function track(get, count, spikeLimit) {
    for (let joint = 0; joint < count; joint++) {
      let segment = [];
      const flush = () => {
        if (segment.length > 2) {
          const result = filterSegment(segment.map(i => get(frames[i])[joint]), segment.map(i => frames[i].time), strength, spikeLimit, stats);
          segment.forEach((index, i) => { get(output[index])[joint] = result[i]; });
        }
        segment = [];
      };
      for (let index = 0; index < frames.length; index++) {
        const point = get(frames[index])?.[joint];
        const previous = segment.at(-1);
        if (!finite(point) || (point.visibility ?? 1) < .4) { flush(); continue; }
        if (previous !== undefined && (frames[index].time - frames[previous].time > .15 || frames[index].time <= frames[previous].time)) flush();
        segment.push(index);
      }
      flush();
    }
  }
  // A learned temporal body already carries a coherent body17/body33 pair.
  // Filtering only body33 would distort limbs relative to the untouched torso.
  track(f => f.body?.pose17 ? undefined : f.body?.landmarks, 33, .08);
  track(f => f.body?.pose17 ? undefined : f.body?.world, 33, .18);
  for (const side of ['Left', 'Right']) {
    track(f => f.hands?.find(hand => hand.side === side)?.landmarks, 21, .08);
    track(f => f.hands?.find(hand => hand.side === side)?.world, 21, .07);
  }
  const nativeFace = frames.some(f => f.face?.topology === 'coco-wholebody-68');
  track(f => f.face?.landmarks, nativeFace ? 68 : 478, .045);
  if (nativeFace) track(f => f.face?.world, 68, .07);
  return {frames: output, stats};
}
