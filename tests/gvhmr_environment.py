"""Opt-in local GPU/timing smoke test. No weights, licensed assets, or accuracy claim.

.venv-gvhmr/Scripts/python.exe tests/gvhmr_environment.py
"""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from backends.gvhmr.diagnostics import prepare_repo, load_inference_modules
from backends.gvhmr.runner import normalize_video, trusted_checkpoint_loading

import cv2
import hydra
import numpy as np
import torch

prepare_repo(ROOT / ".third-party/GVHMR")
cfg = load_inference_modules()
from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np
from hmr4d.utils.preproc.relpose.solver_two_view import CameraParams, TwoPairSolver
model = hydra.utils.instantiate(cfg.network).eval().cuda()
with torch.inference_mode():
    count = 16
    result = model(length=torch.tensor([count], device="cuda"),
                   obs=torch.ones((1, count, 17, 3), device="cuda"),
                   f_cliffcam=torch.zeros((1, count, 3), device="cuda"),
                   f_cam_angvel=torch.zeros((1, count, 6), device="cuda"),
                   f_imgseq=torch.zeros((1, count, 1024), device="cuda"))
    assert result["pred_x"].shape == (1, count, 151)
    assert all(torch.isfinite(value).all().item() for value in result.values())
    torch.cuda.synchronize()
del model, result
torch.cuda.empty_cache()

# Exercise the real installed pycolmap API used for moving-camera estimation.
solver = TwoPairSolver(CameraParams(640, 480, focal_length=600))
rng = np.random.default_rng(42)
points = rng.uniform([-1, -1, 4], [1, 1, 8], (200, 3))
rotation = cv2.Rodrigues(np.array([0., .03, 0.]))[0]
other = points @ rotation.T + np.array([.12, .01, 0.])
project = lambda p: (p[:, :2] / p[:, 2:] * 600 + [320, 240]).astype(np.float32)
estimated = solver.solve(project(points), project(other))
assert np.isfinite(estimated).all()
assert np.max(np.abs(estimated[:3, :3] - rotation)) < .001

base = ROOT / ".gvhmr-runs"
base.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="environment-test-", dir=base) as folder:
    folder = Path(folder)
    source, normalized = folder / "source-24fps.mp4", folder / "normalized-30fps.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24, (128, 128))
    assert writer.isOpened()
    for i in range(48):
        writer.write(np.full((128, 128, 3), i * 4, dtype=np.uint8))
    writer.release()
    info = normalize_video(source, normalized)
    assert info["fps"] == 30 and info["frames"] == 60, info
    assert abs(info["sourceDuration"] - 2) < .001, info
    assert get_video_lwh(str(normalized)) == (60, 128, 128)
    assert read_video_np(str(normalized), scale=.5).shape == (60, 64, 64, 3)
    original_load = torch.load
    with trusted_checkpoint_loading():
        assert torch.load is not original_load
    assert torch.load is original_load

print("GVHMR_ENVIRONMENT_SMOKE_OK: random-weight temporal GPU forward; pycolmap geometry; 24-to-30fps timing/video IO; scoped checkpoint override")
print("No pretrained inference or motion accuracy was tested.")
