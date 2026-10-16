"""Loopback-only studio and authenticated local capture API. No cloud uploads."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import hmac
import json
import mimetypes
import re
import secrets
import shutil
from urllib.parse import unquote, urlsplit

from local_jobs import JobError, JobManager

ROOT = Path(__file__).parent
STATIC_FILES = {"index.html", "styles.css", "root-motion.css", "repair.css", "app.mjs", "local-capture.mjs", "rtmw3d.worker.mjs",
                "core.mjs", "tracker.worker.mjs", "root-motion.mjs", "refinement.mjs",
                "blender_importer.py", "gvhmr.html", "GVHMR Setup.md", "README.md",
                "rtmw3d.html", "RTMW3D.md", "backends/rtmw3d/motion.mjs"}
MAX_VIDEO = 512 * 1024 * 1024
MAX_CAPTURE = 128 * 1024 * 1024


class StudioHandler(SimpleHTTPRequestHandler):
    def _local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed or (origin is not None and origin != "http://" + host) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise JobError("Only this studio's loopback origin is allowed.", 403)

    def _authenticate(self):
        self._local_request()
        token = self.headers.get("X-Framecraft-Token", "")
        if not hmac.compare_digest(token, self.server.api_token):
            raise JobError("Invalid local session token. Refresh the studio.", 403)

    def _json(self, data, status=200):
        encoded = json.dumps(data, allow_nan=False).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _length(self, maximum):
        if self.headers.get("Transfer-Encoding"):
            raise JobError("Chunked requests are not supported.", 400)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise JobError("A Content-Length is required.", 411)
        if not 0 < length <= maximum:
            raise JobError("Request exceeds the local size limit.", 413)
        return length

    def _read_json(self, maximum):
        if self.headers.get_content_type() != "application/json":
            raise JobError("Expected application/json.", 415)
        length = self._length(maximum)
        content = self.rfile.read(length)
        if len(content) != length:
            raise JobError("Incomplete request.")
        try:
            return json.loads(content)
        except (ValueError, UnicodeError) as error:
            raise JobError("Invalid JSON.") from error

    def _file(self, path, filename):
        if not path.is_file(): raise JobError("Result is not ready.", 409)
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if path.suffix == ".json" else "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as source: shutil.copyfileobj(source, self.wfile)

    def _api(self):
        self.connection.settimeout(60)
        self._local_request()
        path = urlsplit(self.path).path
        manager = self.server.jobs
        if self.command == "GET" and path == "/api/session":
            self._json({"token": self.server.api_token, "automaticSetup": True,
                        "installed": (ROOT / ".venv-rtmw3d/Scripts/python.exe").is_file() and (ROOT / ".models/rtmw3d/rtmw3d-x.onnx").is_file(),
                        "blenderAvailable": bool(manager.blender)})
            return
        self._authenticate()
        if self.command == "POST" and path == "/api/jobs":
            job = manager.create(self._read_json(16384))
            self._json(job.snapshot(), 202)
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(video|tracking|export|capture|blend))?", path)
        if not match: raise JobError("Unknown API route.", 404)
        job = manager.get(match[1])
        action = match[2]
        if self.command == "GET" and not action:
            self._json(job.snapshot())
        elif self.command == "DELETE" and not action:
            manager.cancel(job)
            self._json(job.snapshot())
        elif self.command == "PUT" and action == "video":
            if self.headers.get_content_type() != "application/octet-stream": raise JobError("Expected video bytes.", 415)
            remaining = self._length(MAX_VIDEO)
            manager.claim_upload(job)
            try:
                with job.video.open("xb") as output:
                    while remaining:
                        if job.cancelled.is_set(): raise JobError("Capture cancelled.", 409)
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk: raise JobError("Video upload was interrupted.")
                        output.write(chunk)
                        remaining -= len(chunk)
                if job.cancelled.is_set(): raise JobError("Capture cancelled.", 409)
                job.uploaded.set()
            except Exception:
                manager.cancel(job)
                raise
            self._json(job.snapshot(), 202)
        elif self.command == "GET" and action == "tracking":
            if job.snapshot()["state"] != "tracking-ready": raise JobError("Tracking is not ready.", 409)
            self._file(job.tracking, "motion.tracking.json")
        elif self.command == "POST" and action == "export":
            manager.save_capture(job, self._read_json(MAX_CAPTURE))
            self._json(job.snapshot(), 202)
        elif self.command == "GET" and action in ("capture", "blend"):
            snapshot = job.snapshot()
            if not snapshot[action + "Ready"]: raise JobError("Result is not ready.", 409)
            self._file(job.capture if action == "capture" else job.blend, "Framecraft.mocap.json" if action == "capture" else "Framecraft.blend")
        else:
            raise JobError("Unsupported API action.", 405)

    def _dispatch_api(self):
        try:
            self._api()
        except JobError as error:
            self.close_connection = True
            self._json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            self.close_connection = True
            self._json({"error": "Local request failed. Check the studio server window and retry."}, 500)

    def do_GET(self):
        if urlsplit(self.path).path.startswith("/api/"): self._dispatch_api()
        else: super().do_GET()

    def do_POST(self): self._dispatch_api()
    def do_PUT(self): self._dispatch_api()
    def do_DELETE(self): self._dispatch_api()

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        if path not in ("", "/") and path.removeprefix("/") not in STATIC_FILES:
            self.send_error(404, "Not an app file")
            return None
        return super().send_head()

    def end_headers(self):
        if not urlsplit(self.path).path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    mimetypes.add_type("text/javascript", ".mjs")
    handler = partial(StudioHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.api_token = secrets.token_urlsafe(32)
    server.jobs = JobManager(ROOT)
    print(f"Motion Capture Studio: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.close()
        server.server_close()


if __name__ == "__main__":
    main()
