import {buildCapture} from './backends/rtmw3d/motion.mjs';

self.onmessage = ({data}) => {
  try { self.postMessage({capture:buildCapture(data.tracking,data.options)}); }
  catch(error) { self.postMessage({error:error.message}); }
};
