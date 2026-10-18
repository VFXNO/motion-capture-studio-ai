"""Ungated Apache-2.0 models, pinned to published HF revisions and LFS hashes.

No SMPL/SMPL-X assets, accounts, API keys, or pickle checkpoint execution.
"""
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

POSE = {"name": "rtmw3d-x.onnx", "sha256": "4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd",
        "url": "https://huggingface.co/Soykaf/RTMW3D-x/resolve/a4f0d54c52ee44ee4e1dda1936d714490c6234d2/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"}
DETECTOR = {"name": "yolox-m.zip", "sha256": "a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde",
            "url": "https://huggingface.co/Tau-J/RTMPose/resolve/cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"}
BODY2D = {"name": "rtmpose-x-body7.zip", "sha256": "1174dd5ccf780501835017eb7376a93fc202238457549a8af8f6aa6d44ae548a",
          "url": "https://huggingface.co/Tau-J/RTMPose/resolve/cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_one(spec, directory):
    target = directory / spec["name"]
    if target.exists():
        if digest(target) != spec["sha256"]:
            raise ValueError(f"Existing model has an unexpected hash; not overwriting: {target}")
        return target
    print(f"Downloading {spec['name']} (Apache-2.0, no account required)", flush=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".part") as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(spec["url"], timeout=60) as response:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        except Exception:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if digest(temporary) != spec["sha256"]:
            raise ValueError("Downloaded model hash does not match the published LFS hash.")
        # On Windows rename refuses an existing destination, preserving user files.
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def extract_model(archive, detector):
    with zipfile.ZipFile(archive) as handle:
        models = [name for name in handle.namelist() if name.endswith("end2end.onnx")]
        if len(models) != 1:
            raise ValueError("Expected exactly one detector ONNX in the verified archive.")
        # Never extract arbitrary ZIP paths, scripts, or traversal entries.
        with handle.open(models[0]) as source:
            if detector.exists():
                if hashlib.sha256(source.read()).hexdigest() != digest(detector):
                    raise ValueError("Existing extracted detector does not match its verified archive.")
            else:
                with detector.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return detector


def download_body2d(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return extract_model(download_one(BODY2D, directory), directory / "rtmpose-x-body7.onnx")


def download_models(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pose = download_one(POSE, directory)
    detector = extract_model(download_one(DETECTOR, directory), directory / "yolox-m.onnx")
    return pose, detector
