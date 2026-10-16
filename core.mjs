export const FORMAT_VERSION = 1;
export const POSE_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [27,29],[28,30],[29,31],[30,32],[27,31],[28,32]
];
export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]
];

export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00.00';
  const centiseconds = Math.floor(seconds * 100 + 0.00001);
  return `${String(Math.floor(centiseconds / 6000)).padStart(2, '0')}:${String(Math.floor(centiseconds / 100) % 60).padStart(2, '0')}.${String(centiseconds % 100).padStart(2, '0')}`;
}

// Sample strictly before the end; seeking to duration often produces no decoded frame.
export function sampleTimes(duration, fps) {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(fps) || fps <= 0) {
    throw new Error('Video duration and frame rate must be positive, finite numbers.');
  }
  return Array.from({length: Math.ceil(duration * fps)}, (_, i) => i / fps);
}

function smoothPoints(previous, current, amount) {
  if (!current || !previous || current.length !== previous.length) return current;
  return current.map((point, index) => {
    const old = previous[index];
    // Do not drag a reappearing joint toward a low-confidence estimate.
    if ((point.visibility ?? 1) < 0.3 || (old.visibility ?? 1) < 0.3) return point;
    return {...point,
      x: point.x * (1 - amount) + old.x * amount,
      y: point.y * (1 - amount) + old.y * amount,
      z: point.z * (1 - amount) + old.z * amount
    };
  });
}

export function smoothFrame(previous, current, amount) {
  if (!previous || amount <= 0) return current;
  amount = Math.min(0.95, Math.max(0, amount));
  const result = {...current};
  if (current.body && previous.body) result.body = {
    ...current.body,
    landmarks: smoothPoints(previous.body.landmarks, current.body.landmarks, amount),
    world: smoothPoints(previous.body.world, current.body.world, amount)
  };
  if (current.face && previous.face) result.face = {
    ...current.face,
    landmarks: smoothPoints(previous.face.landmarks, current.face.landmarks, amount),
    blendshapes: Object.fromEntries(Object.entries(current.face.blendshapes).map(([name, value]) =>
      [name, value * (1 - amount) + (previous.face.blendshapes[name] ?? value) * amount]))
  };
  result.hands = current.hands.map(hand => {
    const old = previous.hands.find(item => item.side === hand.side);
    return old ? {...hand, landmarks: smoothPoints(old.landmarks, hand.landmarks, amount),
      world: smoothPoints(old.world, hand.world, amount)} : hand;
  });
  return result;
}

export function nearestFrame(frames, time, fps) {
  if (!frames.length) return null;
  const index = Math.min(frames.length - 1, Math.max(0, Math.round(time * fps)));
  const frame = frames[index];
  return Math.abs(frame.time - time) <= 1.5 / fps ? frame : null;
}

export function captureSummary(frames) {
  return {
    frames: frames.length,
    body: frames.filter(frame => frame.body).length,
    hands: frames.filter(frame => frame.hands.length).length,
    face: frames.filter(frame => frame.face).length
  };
}

export function associateHands(hands, body) {
  let result = hands.map(hand => ({...hand}));
  const left = body?.landmarks?.[15], right = body?.landmarks?.[16];
  const reliable = point => point && (point.visibility ?? 1) > 0.5;
  const distance = (hand, wrist) => Math.hypot(hand.landmarks[0].x - wrist.x, hand.landmarks[0].y - wrist.y);
  if (reliable(left) && reliable(right)) {
    if (result.length === 2) {
      const straight = distance(result[0], left) + distance(result[1], right);
      const crossed = distance(result[0], right) + distance(result[1], left);
      result[0].side = straight <= crossed ? 'Left' : 'Right';
      result[1].side = straight <= crossed ? 'Right' : 'Left';
    } else if (result.length === 1) {
      result[0].side = distance(result[0], left) <= distance(result[0], right) ? 'Left' : 'Right';
    }
  }
  // If body association is unavailable, keep the strongest detection per model label.
  return result.filter((hand, index, all) => !all.some((other, otherIndex) =>
    other.side === hand.side && (other.score > hand.score || (other.score === hand.score && otherIndex < index))));
}

export function createCapture(metadata, frames, faceEdges, rootMotion = null) {
  return {format: 'framecraft-mocap', version: FORMAT_VERSION,
    coordinateSystem: 'MediaPipe: x right, y down, z away from camera; body/hand world in meters; face normalized image coordinates',
    metadata, faceEdges, ...(rootMotion ? {rootMotion} : {}), frames};
}
