"""First-use bootstrap. Standard library only; never installs into base Python."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

REVISION = "03a1693e59e4f7cd84582c0fb30459b3bf18ad42"
SOURCE_HASH = "c6db21ed95dc79563ad821533790409a5a9012e48e62b1e32e888a47af0296af"
SOURCE_URL = f"https://codeload.github.com/Tau-J/rtmlib/zip/{REVISION}"


def source_revision(repo):
    repo = Path(repo)
    if (repo / ".git").exists():
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    manifest = json.loads((repo / ".framecraft-source.json").read_text(encoding="utf8"))
    return manifest["revision"] if manifest.get("sha256") == SOURCE_HASH else None


def ensure_source(root):
    repo = Path(root) / ".third-party/rtmlib"
    if repo.exists():
        if source_revision(repo) != REVISION:
            raise ValueError("Existing rtmlib is not the pinned version. It was not overwritten.")
        return repo
    print("Downloading verified RTMW3D support code (no Git installation needed)", flush=True)
    with urlopen(SOURCE_URL, timeout=60) as response:
        archive = response.read(8 * 1024 * 1024 + 1)
    if hashlib.sha256(archive).hexdigest() != SOURCE_HASH:
        raise ValueError("Support-code checksum mismatch; no source files installed.")
    repo.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=repo.parent, prefix="rtmlib-install-") as temporary:
        staging = Path(temporary) / "rtmlib"
        staging.mkdir()
        prefix = f"rtmlib-{REVISION}/"
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            for member in handle.infolist():
                if not member.filename.startswith(prefix):
                    raise ValueError("Unexpected source archive layout.")
                relative = member.filename[len(prefix):]
                target = staging / relative
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise ValueError("Unsafe archive path.")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif relative:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
        (staging / ".framecraft-source.json").write_text(json.dumps({"revision": REVISION, "sha256": SOURCE_HASH}), encoding="utf8")
        if repo.exists():
            raise FileExistsError("Support-code destination appeared during setup; not overwriting it.")
        staging.rename(repo)
    return repo


def environment_ready(root):
    executable = Path(root) / ".venv-rtmw3d" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        return False
    probe = "from importlib.metadata import version; import av,cv2,numpy,onnxruntime,tqdm; assert version('numpy')=='1.26.4'; assert version('onnxruntime-directml')=='1.24.4'"
    try:
        return subprocess.run([str(executable), "-c", probe], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def setup(root):
    root = Path(root)
    if os.name != "nt":
        raise RuntimeError("Automatic DirectML setup currently supports Windows only. Choose browser MediaPipe on other systems.")
    if sys.version_info[:2] != (3, 12):
        try:
            python = subprocess.check_output(["py", "-3.12", "-c", "import sys; print(sys.executable)"], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("Python 3.12 is required for this Windows AI engine. The studio launcher can use an installed Python 3.12.") from error
        subprocess.run([python, str(root / "tools/rtmw3d.py"), "setup"], check=True)
        return
    executable = root / ".venv-rtmw3d/Scripts/python.exe"
    if not executable.exists():
        environment = executable.parent.parent
        marker = environment / ".framecraft-created"
        if environment.exists() and not marker.is_file():
            raise FileExistsError("Incomplete/custom .venv-rtmw3d exists; not replacing user files.")
        environment.mkdir(exist_ok=True)
        marker.write_text("Framecraft isolated RTMW3D environment\n", encoding="utf8")
        print("Creating isolated local AI environment", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    if not environment_ready(root):
        print("Installing local AI dependencies; base PyTorch/ROCm is unchanged", flush=True)
        subprocess.run([str(executable), "-m", "pip", "install", "--disable-pip-version-check", "-r",
                        str(root / "backends/rtmw3d/requirements.txt")], check=True)
    ensure_source(root)
    subprocess.run([str(executable), str(root / "tools/rtmw3d.py"), "download"], check=True)
    print("RTMW3D ready", flush=True)
