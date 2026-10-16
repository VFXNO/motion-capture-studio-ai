"""Bounded, single-worker local AI jobs. No shell commands or caller paths."""
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Event, Lock, Thread
import uuid


class JobError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def find_blender():
    found = shutil.which("blender")
    if found:
        return Path(found)
    foundation = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation"
    candidates = list(foundation.glob("Blender */blender.exe"))
    def version(path):
        return tuple(int(v) for v in re.findall(r"\d+", path.parent.name))
    return max(candidates, key=version) if candidates else None


def validate_options(options):
    if not isinstance(options, dict):
        raise JobError("Expected capture options.")
    filename = options.get("filename", "")
    if not isinstance(filename, str) or not 1 <= len(filename) <= 240 or any(ord(c) < 32 for c in filename):
        raise JobError("Invalid video filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"):
        raise JobError("Use an MP4, MOV, WebM, MKV, or AVI video.")
    for key, lo, hi in (("height", 1, 2.5), ("fov", 30, 100), ("duration", .01, 60)):
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise JobError(f"Invalid {key}; use a video at most 60 seconds long.")
    if type(options.get("fps")) is not int or options["fps"] not in (12, 24, 30):
        raise JobError("Frame rate must be 12, 24, or 30.")
    if options.get("device") not in ("dml", "cpu") or type(options.get("makeBlend")) is not bool:
        raise JobError("Invalid compute/export option.")
    return {key: options[key] for key in ("filename", "height", "fov", "duration", "fps", "device", "makeBlend")}


class Job:
    def __init__(self, root, options):
        self.id = uuid.uuid4().hex
        self.options = options
        self.directory = root / ".rtmw3d-runs" / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.video = self.directory / ("input" + Path(options["filename"]).suffix.lower())
        self.tracking = self.directory / "motion.tracking.json"
        self.capture = self.directory / "motion.mocap.json"
        self.blend = self.directory / "motion.blend"
        self.cancelled, self.uploaded, self.exported = Event(), Event(), Event()
        self.lock = Lock()
        self.process = None
        self.state, self.message, self.progress = "uploading", "Sending video to this PC's local server", 0
        self.receiving = False
        self.logs = deque(maxlen=12)
        self.warning = None

    def update(self, state=None, message=None, progress=None):
        with self.lock:
            if self.cancelled.is_set() and state != "cancelled":
                return
            if state is not None: self.state = state
            if message is not None: self.message = message
            if progress is not None: self.progress = progress

    def snapshot(self):
        with self.lock:
            return {"id": self.id, "state": self.state, "message": self.message, "progress": self.progress,
                    "warning": self.warning, "blendReady": self.state == "complete" and self.blend.is_file(),
                    "captureReady": self.state == "complete" and self.capture.is_file()}


class JobManager:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = Lock()
        self.jobs, self.active = {}, None
        self.blender = find_blender()

    def create(self, options):
        options = validate_options(options)
        if options["makeBlend"] and not self.blender:
            raise JobError("Blender is not installed/detected. Turn off 'Create Blender scene' to capture JSON.")
        with self.lock:
            if self.active is not None:
                raise JobError("Another local capture is running. Cancel it or wait for it to finish.", 409)
            if len(self.jobs) >= 100:
                raise JobError("This session has reached 100 jobs. Restart the studio; saved files are retained.", 409)
            job = Job(self.root, options)
            self.jobs[job.id], self.active = job, job.id
        Thread(target=self._worker, args=(job,), daemon=True).start()
        return job

    def get(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
        if not job:
            raise JobError("Unknown local capture.", 404)
        return job

    def claim_upload(self, job):
        with job.lock:
            if job.state != "uploading" or job.receiving or job.cancelled.is_set():
                raise JobError("This job cannot receive a video now.", 409)
            job.receiving = True

    def save_capture(self, job, data):
        from blender_importer import validate_capture
        with job.lock:
            if job.state != "tracking-ready" or job.cancelled.is_set():
                raise JobError("This job is not ready for export.", 409)
            if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict) or data["metadata"].get("engine") != "rtmw3d-x":
                raise JobError("Expected an RTMW3D capture.")
            try:
                validate_capture(data)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise JobError(f"Invalid capture: {error}") from error
            native = json.loads(job.tracking.read_text(encoding="utf8"))
            if len(data["frames"]) != len(native["frames"]) or data["metadata"]["fps"] != job.options["fps"]:
                raise JobError("Capture sample count/rate does not match this job.")
            data["metadata"]["filename"] = job.options["filename"]
            with job.capture.open("x", encoding="utf8") as handle:
                json.dump(data, handle, allow_nan=False, separators=(",", ":"))
            job.state, job.message = "exporting", "Creating your Blender scene" if job.options["makeBlend"] else "Saving capture"
        job.exported.set()

    def _stop(self, process):
        if process and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            else:
                process.terminate()

    def cancel(self, job):
        with job.lock:
            if job.state in ("complete", "failed", "cancelled"):
                return
            job.cancelled.set()
            process = job.process
            job.state, job.message = "cancelled", "Capture cancelled; no partial result is offered"
        self._stop(process)
        job.uploaded.set()
        job.exported.set()

    def _command(self, job, arguments):
        if job.cancelled.is_set(): raise InterruptedError()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen([str(arg) for arg in arguments], cwd=self.root, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf8", errors="replace", creationflags=flags)
        with job.lock:
            job.process = process
        if job.cancelled.is_set(): self._stop(process)
        try:
            with (job.directory / "process.log").open("a", encoding="utf8") as log:
                for line in process.stdout:
                    clean = line.replace("\x00", "").strip()
                    log.write(clean + "\n")
                    if clean:
                        job.logs.append(clean[-1000:])
                    match = re.search(r"RTMW3D (\d+)/(\d+)", clean)
                    if match and not job.cancelled.is_set():
                        index, count = map(int, match.groups())
                        job.update(message=f"Tracking frame {index} of {count}", progress=.1 + .7 * index / max(1, count))
                    elif clean.startswith(("Downloading", "Installing", "Creating isolated", "RTMW3D ready")):
                        job.update(message=clean[:240])
            code = process.wait()
        finally:
            process.stdout.close()
            with job.lock:
                job.process = None
        if job.cancelled.is_set(): raise InterruptedError()
        if code:
            raise RuntimeError("Local engine failed: " + "\n".join(list(job.logs)[-4:]))

    def _worker(self, job):
        try:
            if not job.uploaded.wait(600):
                raise RuntimeError("Video upload timed out. Choose the video and retry.")
            if job.cancelled.is_set(): raise InterruptedError()
            job.update("preparing", "Preparing local AI automatically; first use downloads models", .02)
            self._command(job, [sys.executable, "-u", self.root / "tools/rtmw3d.py", "setup"])
            job.update("tracking", "Running RTMW3D-X locally", .1)
            python = self.root / ".venv-rtmw3d" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            self._command(job, [python, "-u", self.root / "tools/rtmw3d.py", "run", job.video, job.tracking,
                                "--fps", job.options["fps"], "--height", job.options["height"],
                                "--fov", job.options["fov"], "--device", job.options["device"]])
            job.update("tracking-ready", "Building body, fingers, face and estimated floor travel", .82)
            if not job.exported.wait(600):
                raise RuntimeError("The browser disconnected before export. Retry with the studio tab open.")
            if job.cancelled.is_set(): raise InterruptedError()
            if job.options["makeBlend"]:
                job.update(message="Creating your Blender scene in a fresh background process", progress=.9)
                try:
                    self._command(job, [self.blender, "--background", "--factory-startup", "--python-exit-code", "1",
                                        "--python", self.root / "tools/build-blender-scene.py", "--", job.capture, job.blend])
                    if not job.blend.is_file(): raise RuntimeError("Blender did not create a scene.")
                except InterruptedError:
                    raise
                except Exception as error:
                    job.warning = f"Capture succeeded, but Blender export failed: {error}"
            job.update("complete", "Capture and Blender scene are ready" if job.blend.is_file() else "Capture is ready", 1)
        except InterruptedError:
            job.update("cancelled", "Capture cancelled; no partial result is offered")
        except Exception as error:
            job.update("failed", str(error)[-3000:])
        finally:
            with self.lock:
                if self.active == job.id: self.active = None

    def close(self):
        for job in list(self.jobs.values()):
            self.cancel(job)
