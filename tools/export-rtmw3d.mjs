import {readFile,writeFile} from 'node:fs/promises';
import {buildCapture} from '../backends/rtmw3d/motion.mjs';
const [input,output,pitch='10',cleanup='.25']=process.argv.slice(2);
try {
  if(!input||!output) throw new Error('Usage: node tools/export-rtmw3d.mjs INPUT.tracking.json NEW.mocap.json [PITCH=10] [CLEANUP=.25]');
  const capture=buildCapture(JSON.parse(await readFile(input,'utf8')),{pitch:Number(pitch),cleanupStrength:Number(cleanup)});
  await writeFile(output,JSON.stringify(capture,(_key,value)=>typeof value==='number'?Math.round(value*1e6)/1e6:value),{flag:'wx'});
  console.log(JSON.stringify({output,engine:capture.metadata.engine,frames:capture.frames.length,
    body:capture.frames.filter(f=>f.body).length,hands:capture.frames.filter(f=>f.hands.length).length,
    face:capture.frames.filter(f=>f.face).length,root:capture.rootMotion,
    calibration:capture.metadata.calibration},null,2));
} catch(error) {console.error(error.message);process.exitCode=1;}
