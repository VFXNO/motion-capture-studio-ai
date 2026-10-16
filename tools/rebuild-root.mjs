// Repair saved captures without re-running the AI models. Never overwrite input.
import {readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {rebuildCaptureRoot} from '../root-motion.mjs';

const [input, output, fov = '60', pitch = '10', cleanupStrength = '0'] = process.argv.slice(2);
try {
  if (!input || !output) throw new Error('Usage: node tools/rebuild-root.mjs INPUT.mocap.json OUTPUT.mocap.json [FOV=60] [PITCH=10] [CLEANUP=0]');
  if (resolve(input).toLowerCase() === resolve(output).toLowerCase()) throw new Error('Choose a new output file; the original is never overwritten.');
  const original = JSON.parse(await readFile(input, 'utf8'));
  const repaired = rebuildCaptureRoot(original, {fov: Number(fov), pitch: Number(pitch), stabilize: true, cleanupStrength: Number(cleanupStrength)});
  await writeFile(output, JSON.stringify(repaired, (_key, value) => typeof value === 'number'
    ? Math.round(value * 1e6) / 1e6 : value), {encoding: 'utf8', flag: 'wx'});
  const positions = repaired.frames.filter(f => f.root?.status === 'tracked').map(f => f.root.position);
  const spans = Object.fromEntries(['x', 'y', 'z'].map(axis => [axis,
    Math.max(...positions.map(p => p[axis])) - Math.min(...positions.map(p => p[axis]))]));
  console.log(JSON.stringify({output: resolve(output), samples: positions.length, spansMeters: spans}, null, 2));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
