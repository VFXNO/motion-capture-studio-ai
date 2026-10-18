// Approximate monocular root reconstruction. Assumes a fixed, level-roll camera,
// a flat floor, and MediaPipe's body scale. This is not camera-motion estimation.
import {refineCaptureFrames} from './refinement.mjs';
const JOINTS = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const validPoint = p => p && ['x', 'y', 'z'].every(axis => Number.isFinite(p[axis]));
const visible = p => validPoint(p) && (p.visibility ?? 1) >= 0.5;
const add = (a, b) => ({x: a.x + b.x, y: a.y + b.y, z: a.z + b.z});
const subtract = (a, b) => ({x: a.x - b.x, y: a.y - b.y, z: a.z - b.z});
const mix = (a, b, alpha) => Object.fromEntries(['x', 'y', 'z'].map(axis => [axis, a[axis] + (b[axis] - a[axis]) * alpha]));
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export function cameraToGround(point, pitchDegrees = 0) {
  const pitch = pitchDegrees * Math.PI / 180;
  return {x: point.x,
    y: Math.cos(pitch) * point.z - Math.sin(pitch) * point.y,
    z: -Math.sin(pitch) * point.z - Math.cos(pitch) * point.y};
}

export function hipCenter(body) {
  if (!validPoint(body?.world?.[23]) || !validPoint(body?.world?.[24])) return null;
  return mix(body.world[23], body.world[24], 0.5);
}

export function estimateCameraTranslation(body, {width, height, fov = 60}) {
  if(body?.pose17) return estimateOrthographicTranslation(body,{width,height,fov});
  const center = hipCenter(body);
  if (!center || !visible(body.landmarks?.[23]) || !visible(body.landmarks?.[24])) return null;
  const focal = width / (2 * Math.tan(fov * Math.PI / 360));
  const rows = JOINTS.flatMap(index => {
    const image = body.landmarks?.[index], world = body.world?.[index];
    if (!visible(image) || !validPoint(world) || image.x < 0 || image.x > 1 || image.y < 0 || image.y > 1) return [];
    const local = subtract(world, center);
    const u = (image.x - 0.5) * width / focal;
    const v = (image.y - 0.5) * height / focal;
    return [{u, v, bx: u * local.z - local.x, by: v * local.z - local.y,
      weight: image.visibility ?? 1, baseWeight: image.visibility ?? 1, local}];
  });
  if (rows.length < 6) return null;
  function fit() {
    const weight = rows.reduce((sum, row) => sum + row.weight, 0);
    const mean = key => rows.reduce((sum, row) => sum + row[key] * row.weight, 0) / weight;
    const u = mean('u'), v = mean('v'), bx = mean('bx'), by = mean('by');
    let numerator = 0, denominator = 0;
    for (const row of rows) {
      numerator += row.weight * ((row.u - u) * (row.bx - bx) + (row.v - v) * (row.by - by));
      denominator += row.weight * ((row.u - u) ** 2 + (row.v - v) ** 2);
    }
    if (denominator / weight < 0.0001) return null;
    const z = -numerator / denominator;
    return {x: bx + u * z, y: by + v * z, z};
  }
  let position = fit();
  // Reweight inconsistent joints instead of letting one bad elbow/ankle pull the
  // entire person sideways or change the inferred depth (Huber-style IRLS).
  for (let iteration = 0; iteration < 5; iteration++) {
    if (!validPoint(position) || position.z < .4 || position.z > 50) return null;
    const residuals = rows.map(row => Math.hypot(
      (row.local.x + position.x) / (row.local.z + position.z) - row.u,
      (row.local.y + position.y) / (row.local.z + position.z) - row.v));
    const threshold = Math.max(.004, 2.5 * median(residuals));
    rows.forEach((row, i) => { row.weight = row.baseWeight * Math.min(1, threshold / Math.max(residuals[i], 1e-9)); });
    position = fit();
  }
  if (!validPoint(position) || position.z < .4 || position.z > 50) return null;
  const z = position.z;
  const weight = rows.reduce((sum, row) => sum + row.weight, 0);
  let error = 0;
  for (const row of rows) {
    const depth = row.local.z + z;
    if (depth <= 0.05) return null;
    error += row.weight * (((row.local.x + position.x) / depth - row.u) ** 2 +
      ((row.local.y + position.y) / depth - row.v) ** 2);
  }
  const rms = Math.sqrt(error / weight);
  if (rms > 0.08) return null;
  return {position, confidence: Math.max(0, 1 - rms / 0.08) * Math.min(1, rows.length / 10)};
}

export function estimateOrthographicTranslation(body,{width,height,fov=60}) {
  // Temporal lifting predicts camera-relative joints under scaled orthographic
  // training projections. Fitting perspective directly exaggerates limb depth.
  const center=hipCenter(body);
  if(!center) return null;
  const rows=JOINTS.flatMap(index=>{
    const p=body.world[index],q=body.landmarks[index];
    return visible(q)&&validPoint(p)?[{x:p.x-center.x,y:p.y-center.y,u:q.x*width,v:q.y*height,
      weight:q.visibility??1,base:q.visibility??1}]:[];
  });
  if(rows.length<6) return null;
  let solution;
  for(let pass=0;pass<4;pass++) {
    const total=rows.reduce((s,r)=>s+r.weight,0);
    const average=key=>rows.reduce((s,r)=>s+r[key]*r.weight,0)/total;
    const x=average('x'),y=average('y'),u=average('u'),v=average('v');
    const denominator=rows.reduce((s,r)=>s+r.weight*((r.x-x)**2+(r.y-y)**2),0);
    if(denominator<1e-6) return null;
    const scale=rows.reduce((s,r)=>s+r.weight*((r.x-x)*(r.u-u)+(r.y-y)*(r.v-v)),0)/denominator;
    if(!Number.isFinite(scale)||scale<10) return null;
    solution={scale,tx:u-scale*x,ty:v-scale*y};
    const errors=rows.map(r=>Math.hypot(r.u-(scale*r.x+solution.tx),r.v-(scale*r.y+solution.ty)));
    const threshold=Math.max(4,2.5*median(errors));
    rows.forEach((r,i)=>r.weight=r.base*Math.min(1,threshold/Math.max(1e-9,errors[i])));
  }
  const {scale,tx,ty}=solution;
  const focal=width/(2*Math.tan(fov*Math.PI/360));
  const position={x:(tx-width/2)/scale,y:(ty-height/2)/scale,z:focal/scale};
  if(position.z<.4||position.z>50) return null;
  const rms=Math.sqrt(rows.reduce((s,r)=>s+r.weight*((r.u-scale*r.x-tx)**2+(r.v-scale*r.y-ty)**2),0)/rows.reduce((s,r)=>s+r.weight,0))/Math.max(width,height);
  if(rms>.08) return null;
  return {position,confidence:Math.max(0,1-rms/.08),pixelsPerMeter:scale,cameraModel:'scaled-orthographic'};
}

function localFeet(body, pitch) {
  const center = hipCenter(body);
  const result = {};
  if (!center) return result;
  for (const [side, heel, toe] of [['Left', 29, 31], ['Right', 30, 32]]) {
    if (!visible(body.landmarks?.[heel]) || !visible(body.landmarks?.[toe]) ||
      !validPoint(body.world?.[heel]) || !validPoint(body.world?.[toe])) continue;
    const a = cameraToGround(subtract(body.world[heel], center), pitch);
    const b = cameraToGround(subtract(body.world[toe], center), pitch);
    result[side] = {...mix(a, b, 0.5), z: Math.min(a.z, b.z)};
  }
  return result;
}

export function reconstructRootMotion(frames, options) {
  const {width, height, fov = 60, pitch = 10, stabilize = true} = options;
  if (![width, height, fov, pitch].every(Number.isFinite) || width <= 0 || height <= 0 ||
    fov < 30 || fov > 100 || pitch < -15 || pitch > 60) throw new Error('Invalid root camera settings.');
  const info = {enabled: false, method: 'grounded-perspective-v1', fov, pitch, stabilize,
    coordinates: 'Blender Z-up, X camera-right, Y away along floor; estimated meters; Z is hip height above floor',
    assumptions: 'Fixed camera, flat floor, no camera roll. Scale and depth are monocular estimates.'};
  if(frames.some(f=>f.body?.pose17)) {
    info.cameraModel='scaled-orthographic';
    info.assumptions='Temporal-body scaled-orthographic translation; depth estimated from FOV and projected size. Fixed camera, flat floor; not learned world travel.';
  }
  const raw = frames.map(frame => {
    if (!frame.body) return null;
    const solved = estimateCameraTranslation(frame.body, {width, height, fov});
    return solved ? {hip: cameraToGround(solved.position, pitch), feet: localFeet(frame.body, pitch),
      confidence: solved.confidence} : null;
  });
  // Reject isolated projection spikes without smoothing real constant-velocity travel.
  const records = raw.map((record, index) => {
    if (!record || !raw[index - 1] || !raw[index + 1] || frames[index + 1].time - frames[index - 1].time > 0.3) return record;
    return {...record, hip: Object.fromEntries(['x', 'y', 'z'].map(axis =>
      [axis, median([raw[index - 1].hip[axis], record.hip[axis], raw[index + 1].hip[axis]])]))};
  });
  const floors = records.flatMap(record => record && Object.keys(record.feet).length
    ? [Math.min(...Object.values(record.feet).map(foot => record.hip.z + foot.z))] : []);
  if (!floors.length) return {frames, info: {...info, reason: 'No reliable visible feet. Root motion was not estimated.'}};
  const sorted = floors.sort((a, b) => a - b);
  const floor = sorted[Math.floor((sorted.length - 1) * 0.1)];
  const first = records.find(record => record && Object.keys(record.feet).length);
  const origin = {...first.hip};
  Object.assign(info, {enabled: true, groundHeight: floor, origin,
    validFrames: records.filter(Boolean).length, totalFrames: frames.length});
  let lastHip = null, lastTime = null, lastRoot = null, previousFeet = {}, anchors = {};
  let lastIndex = -2;
  const result = frames.map((frame, index) => {
    const record = records[index];
    if (!record) {
      anchors = {};
      previousFeet = {};
      lastHip = null;
      return {...frame, root: lastRoot ? {...lastRoot, status: 'held', confidence: 0, contacts: {Left: false, Right: false}} : null};
    }
    const dt = lastTime === null ? 0 : frame.time - lastTime;
    const continuous = index === lastIndex + 1 && dt > 0 && dt < 0.3;
    let hip = continuous && lastHip ? mix(lastHip, record.hip, 1 - Math.exp(-dt / 0.06)) : {...record.hip};
    lastHip = {...hip};
    lastTime = frame.time;
    lastIndex = index;
    if (!continuous) { anchors = {}; previousFeet = {}; }
    const absoluteFeet = Object.fromEntries(Object.entries(record.feet).map(([side, foot]) => [side, add(hip, foot)]));
    const contacts = {Left: false, Right: false};
    for (const [side, foot] of Object.entries(absoluteFeet)) {
      const previous = previousFeet[side];
      const speed = previous && continuous ? Math.hypot(foot.x - previous.x, foot.y - previous.y, foot.z - previous.z) / dt : 0;
      contacts[side] = foot.z - floor <= 0.09 && foot.z - floor >= -0.2 && speed < 0.2;
      if (!contacts[side]) delete anchors[side];
    }
    if (stabilize) {
      const candidates = Object.entries(anchors).flatMap(([side, anchor]) => {
        if (!contacts[side] || !record.feet[side]) return [];
        const candidate = subtract(anchor, record.feet[side]);
        // Release inconsistent locks rather than permanently pinning a moving foot.
        if (Math.hypot(candidate.x - hip.x, candidate.y - hip.y) > 0.25) { delete anchors[side]; return []; }
        return [candidate];
      });
      if (candidates.length) {
        const average = axis => candidates.reduce((sum, candidate) => sum + candidate[axis], 0) / candidates.length;
        hip.x = hip.x * 0.05 + average('x') * 0.95;
        hip.y = hip.y * 0.05 + average('y') * 0.95;
      }
      const support = Object.entries(record.feet).filter(([side]) => contacts[side]).map(([, foot]) => foot.z);
      if (support.length) hip.z = floor - Math.min(...support);
      // Keep tracked soles from penetrating the floor; airborne samples retain height.
      const lowest = Math.min(...Object.values(record.feet).map(foot => hip.z + foot.z));
      if (Number.isFinite(lowest) && lowest < floor) hip.z += floor - lowest;
      for (const [side, foot] of Object.entries(record.feet)) {
        if (contacts[side] && !anchors[side]) anchors[side] = add(hip, foot);
      }
    }
    previousFeet = absoluteFeet;
    lastRoot = {position: {x: hip.x - origin.x, y: hip.y - origin.y, z: hip.z - floor},
      confidence: record.confidence, contacts, status: 'tracked'};
    return {...frame, root: lastRoot};
  });
  return {frames: result, info};
}

export function rebuildCaptureRoot(capture, options = {}) {
  if (options.cleanupStrength !== undefined && (!Number.isFinite(options.cleanupStrength) || options.cleanupStrength < 0 || options.cleanupStrength > .85)) {
    throw new Error('Cleanup strength must be 0–0.85.');
  }
  if (capture?.format !== 'framecraft-mocap' || capture.version !== 1 ||
    !Array.isArray(capture.frames) || !capture.frames.length || capture.frames.length > 1800) {
    throw new Error('Choose a Framecraft version 1 capture with 1–1,800 samples.');
  }
  const {width, height} = capture.metadata ?? {};
  // Remove any previous root samples first. An unsuccessful rebuild must never
  // leave stale travel from another camera configuration attached to the file.
  let source = capture.frames.map(({root: _oldRoot, ...frame}) => frame);
  let refinement = capture.metadata?.refinement;
  if (options.cleanupStrength > 0) {
    const cleaned = refineCaptureFrames(source, {strength: options.cleanupStrength});
    source = cleaned.frames;
    refinement = cleaned.stats;
  }
  const {frames, info} = reconstructRootMotion(source, {width, height, ...options});
  if (!info.enabled) throw new Error(info.reason);
  return {...capture, frames, rootMotion: info, metadata: {...capture.metadata,
    rootRebuiltAt: new Date().toISOString(), ...(refinement ? {refinement} : {}),
    notes: 'Root rebuilt from saved pose landmarks. Approximate fixed-camera, flat-ground travel; not camera-motion compensated.'}};
}
