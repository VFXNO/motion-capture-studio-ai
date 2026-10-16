"""Local HTTP safety/lifecycle tests. Fake inference never installs/downloads."""
import functools
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import tempfile
from threading import Thread
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from local_jobs import Job, JobManager
from serve import StudioHandler, ROOT


OPTIONS = {"filename":"dance.mp4","fps":24,"height":1.7,"fov":60,"duration":1,"device":"cpu","makeBlend":False}


class FakeManager(JobManager):
    def _worker(self, job):
        try:
            job.uploaded.wait(3)
            if job.cancelled.is_set(): return
            job.tracking.write_text(json.dumps({"frames":[{}]}),encoding="utf8")
            job.update("tracking-ready","Fake inference finished",.82)
            job.exported.wait(3)
            if not job.cancelled.is_set() and job.capture.exists(): job.update("complete","Ready",1)
        finally:
            with self.lock:
                if self.active==job.id: self.active=None


class QuietHandler(StudioHandler):
    def log_message(self,*args): pass


class LocalAPITest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.server=ThreadingHTTPServer(("127.0.0.1",0),functools.partial(QuietHandler,directory=str(ROOT)))
        self.server.jobs=FakeManager(Path(self.temporary.name))
        self.server.api_token=secrets.token_urlsafe(32)
        self.thread=Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base=f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.jobs.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def call(self,path,method="GET",body=None,token=True,headers=None):
        headers=dict(headers or {})
        if token: headers["X-Framecraft-Token"]=self.server.api_token
        if body is not None and not isinstance(body,bytes):
            body=json.dumps(body).encode()
            headers.setdefault("Content-Type","application/json")
        with urlopen(Request(self.base+path,data=body,method=method,headers=headers),timeout=5) as response:
            return response.status,json.loads(response.read())

    def assert_http(self,code,*args,**kwargs):
        with self.assertRaises(HTTPError) as caught: self.call(*args,**kwargs)
        self.assertEqual(caught.exception.code,code)
        caught.exception.close()

    def create(self):
        code,job=self.call("/api/jobs","POST",OPTIONS)
        self.assertEqual(code,202)
        return job["id"]

    def wait_state(self,identifier,state):
        for _ in range(100):
            _,data=self.call(f"/api/jobs/{identifier}")
            if data["state"]==state: return data
            time.sleep(.01)
        self.fail(f"Job did not reach {state}")

    def test_session_is_local_and_mutations_require_token(self):
        _,session=self.call("/api/session",token=False)
        self.assertEqual(session["token"],self.server.api_token)
        self.assert_http(403,"/api/jobs","POST",OPTIONS,token=False)
        self.assertEqual(self.server.jobs.jobs,{})
        self.assert_http(403,"/api/session",token=False,headers={"Origin":"https://evil.example"})
        self.assert_http(403,"/api/session",token=False,headers={"Host":"evil.example"})
        self.assert_http(403,"/api/session",token=False,headers={"Sec-Fetch-Site":"cross-site"})

    def test_invalid_options_and_oversize_upload_are_rejected(self):
        for change in ({"fps":240},{"height":float('nan')},{"filename":"x.exe"},{"duration":61},{"device":"cuda;whoami"}):
            self.assert_http(400,"/api/jobs","POST",{**OPTIONS,**change})
        identifier=self.create()
        self.assert_http(413,f"/api/jobs/{identifier}/video","PUT",b"x",headers={"Content-Type":"application/octet-stream","Content-Length":str(513*1024*1024)})
        self.call(f"/api/jobs/{identifier}","DELETE")

    def test_only_one_job_and_no_input_or_result_path_access(self):
        identifier=self.create()
        self.assert_http(409,"/api/jobs","POST",OPTIONS)
        for path in (f"/.rtmw3d-runs/{identifier}/input.mp4",f"/api/jobs/{identifier}/input",f"/api/jobs/{identifier}/blend"):
            self.assert_http(409 if path.endswith('/blend') else 404,path)
        self.call(f"/api/jobs/{identifier}","DELETE")

    def test_upload_export_and_private_result_download(self):
        identifier=self.create()
        route=f"/api/jobs/{identifier}"
        self.call(route+"/video","PUT",b"a local video",headers={"Content-Type":"application/octet-stream"})
        self.wait_state(identifier,"tracking-ready")
        _,native=self.call(route+"/tracking")
        self.assertEqual(len(native["frames"]),1)
        self.assert_http(409,route+"/video","PUT",b"replacement",headers={"Content-Type":"application/octet-stream"})
        job=self.server.jobs.get(identifier)
        self.assertEqual(job.video.read_bytes(),b"a local video")
        invalid={"format":"framecraft-mocap","version":1,"metadata":{"engine":"rtmw3d-x","fps":24,"width":640,"height":480},"frames":[]}
        self.assert_http(400,route+"/export","POST",invalid)
        capture={**invalid,"frames":[{"time":0,"body":None,"hands":[],"face":None}]}
        self.call(route+"/export","POST",capture)
        snapshot=self.wait_state(identifier,"complete")
        self.assertTrue(snapshot["captureReady"])
        self.assertFalse(snapshot["blendReady"])
        _,result=self.call(route+"/capture")
        self.assertEqual(result["metadata"]["filename"],"dance.mp4")
        self.assert_http(409,route+"/export","POST",capture)
        self.assert_http(403,route+"/capture",token=False)

    def test_cancel_never_offers_partial_capture_and_releases_worker(self):
        identifier=self.create()
        self.call(f"/api/jobs/{identifier}","DELETE")
        self.wait_state(identifier,"cancelled")
        self.assert_http(409,f"/api/jobs/{identifier}/capture")
        for _ in range(100):
            if self.server.jobs.active is None: break
            time.sleep(.01)
        self.assertIsNone(self.server.jobs.active)
        second=self.create()
        self.assertNotEqual(identifier,second)
        self.call(f"/api/jobs/{second}","DELETE")

    def test_cancel_stops_only_the_created_child_process(self):
        manager=JobManager(Path(self.temporary.name))
        job=Job(manager.root,OPTIONS)
        errors=[]
        def run():
            try: manager._command(job,[sys.executable,"-u","-c","import time; print('READY',flush=True); time.sleep(60)"])
            except InterruptedError: errors.append("cancelled")
        thread=Thread(target=run,daemon=True)
        thread.start()
        for _ in range(200):
            if job.process is not None: break
            time.sleep(.01)
        process=job.process
        self.assertIsNotNone(process)
        manager.cancel(job)
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(process.poll())
        self.assertEqual(errors,["cancelled"])


if __name__=="__main__": unittest.main()
