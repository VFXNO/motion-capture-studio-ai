// Same-origin loopback API only. Models can download; video never goes to them.
let sessionPromise;
export function localSession() {
  return sessionPromise ??= fetch('/api/session', {cache:'no-store'}).then(async response => {
    if(!response.ok) throw new Error('Restart Framecraft with Launch Studio.bat to enable automatic local capture.');
    const session=await response.json();
    if(typeof session.token!=='string'||session.token.length<32) throw new Error('Invalid local studio session. Refresh the page.');
    return session;
  }).catch(error=>{sessionPromise=null;throw error;});
}

async function request(path, {signal,...options}={}) {
  const session=await localSession();
  const response=await fetch(path,{...options,signal,cache:'no-store',headers:{'X-Framecraft-Token':session.token,...options.headers}});
  if(!response.ok) {
    let message=`Local capture request failed (${response.status}).`;
    try {message=(await response.json()).error||message;} catch {}
    throw new Error(message);
  }
  return response;
}

function pause(signal) {
  return new Promise((resolve,reject)=>{
    if(signal.aborted) return reject(new DOMException('Cancelled.','AbortError'));
    const abort=()=>{clearTimeout(timer);reject(new DOMException('Cancelled.','AbortError'));};
    const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},400);
    signal.addEventListener('abort',abort,{once:true});
  });
}

export async function captureLocalVideo({file,options,signal,onProgress,buildCapture}) {
  let job=null,complete=false;
  const cancel=()=>{if(job&&!complete) request(`/api/jobs/${job.id}`,{method:'DELETE',keepalive:true}).catch(()=>{});};
  signal.addEventListener('abort',cancel,{once:true});
  try {
    // Do not abort this tiny reservation request: always learn its ID so Cancel
    // can release a job even when clicked between reservation and video upload.
    job=await (await request('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(options)})).json();
    if(signal.aborted) throw new DOMException('Cancelled.','AbortError');
    onProgress({message:"Sending video only to this PC's local server",progress:0});
    await request(`/api/jobs/${job.id}/video`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:file,signal});
    let exported=false;
    for(;;) {
      if(signal.aborted) throw new DOMException('Cancelled.','AbortError');
      const state=await (await request(`/api/jobs/${job.id}`,{signal})).json();
      onProgress(state);
      if(state.state==='failed') throw new Error(state.message);
      if(state.state==='cancelled') throw new DOMException('Cancelled.','AbortError');
      if(state.state==='tracking-ready'&&!exported) {
        const tracking=await (await request(`/api/jobs/${job.id}/tracking`,{signal})).json();
        // Let the progress message paint before synchronous geometry fitting.
        await pause(signal);
        const capture=await buildCapture(tracking);
        if(signal.aborted) throw new DOMException('Cancelled.','AbortError');
        await request(`/api/jobs/${job.id}/export`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(capture),signal});
        exported=true;
      }
      if(state.state==='complete') {
        const capture=await (await request(`/api/jobs/${job.id}/capture`,{signal})).json();
        complete=true;
        return {capture,job:state};
      }
      await pause(signal);
    }
  } finally {
    signal.removeEventListener('abort',cancel);
    if(job&&!complete) {
      try {await request(`/api/jobs/${job.id}`,{method:'DELETE',keepalive:true});} catch {}
    }
  }
}

export async function blenderSceneBlob(jobID) {
  if(!/^[a-f0-9]{32}$/.test(jobID)) throw new Error('Invalid local scene.');
  return (await request(`/api/jobs/${jobID}/blend`)).blob();
}

export function fitNativeTracking(tracking,options,signal) {
  return new Promise((resolve,reject)=>{
    if(signal.aborted) return reject(new DOMException('Cancelled.','AbortError'));
    const worker=new Worker(new URL('./rtmw3d.worker.mjs',import.meta.url),{type:'module'});
    const cleanup=()=>{worker.terminate();signal.removeEventListener('abort',abort);};
    const abort=()=>{cleanup();reject(new DOMException('Cancelled.','AbortError'));};
    worker.onmessage=({data})=>{
      cleanup();
      if(data.error) reject(new Error(data.error)); else resolve(data.capture);
    };
    worker.onerror=event=>{event.preventDefault();cleanup();reject(new Error(event.message||'Local motion fitting failed.'));};
    signal.addEventListener('abort',abort,{once:true});
    worker.postMessage({tracking,options});
  });
}
