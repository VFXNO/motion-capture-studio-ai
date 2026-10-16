"""Readiness checks never download weights, unpickle assets, or run inference."""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

REVISION = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"
ASSETS = [
    ("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", "SMPL-X neutral NPZ; obtain under your own license at https://smpl-x.is.tue.mpg.de/"),
    ("inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt", "GVHMR release checkpoint"),
    ("inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt", "HMR2.0a feature checkpoint (not HMR2.0b)"),
    ("inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth", "ViTPose huge multi-COCO checkpoint"),
    ("inputs/checkpoints/yolo/yolov8x.pt", "YOLOv8x detector checkpoint"),
]
MODULES = ["torch", "torchvision", "pytorch3d.transforms", "pytorch_lightning", "hydra",
           "hydra_zen", "timm", "smplx", "cv2", "einops", "imageio", "av", "imageio_ffmpeg",
           "ffmpeg", "ultralytics", "lap", "pycolmap", "wis3d", "yacs", "scipy", "skimage", "iopath"]


def prepare_repo(repo):
    repo = Path(repo).resolve()
    if not (repo / "hmr4d/__init__.py").is_file():
        raise ValueError(f"GVHMR source is missing: {repo}. Run tools/setup-gvhmr.ps1 first.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def load_inference_modules():
    # Register only inference configs. Upstream register_store_gvhmr also imports
    # training datasets, metrics and compiled rendering dependencies we don't use.
    for name in ("hmr4d.model.gvhmr.gvhmr_pl_demo", "hmr4d.model.gvhmr.utils.endecoder",
                 "hmr4d.network.gvhmr.relative_transformer", "hmr4d.model.gvhmr.pipeline.gvhmr_pipeline",
                 "hmr4d.utils.preproc"):
        importlib.import_module(name)
    from hydra import compose, initialize_config_module
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        return compose(config_name="demo")


def diagnose(repo, probe=False):
    repo = Path(repo).resolve()
    report = {"engine": "GVHMR", "python": sys.executable, "repo": str(repo),
              "expectedRevision": REVISION, "license": "Research/non-commercial; see upstream LICENSE and asset terms",
              "missingModules": [], "assets": [], "errors": [], "probed": probe,
              "inferenceVerified": False}
    if not (repo / "hmr4d/__init__.py").is_file():
        report["errors"].append("GVHMR source not installed.")
    else:
        try:
            report["revision"] = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
            if report["revision"] != REVISION:
                report["errors"].append("GVHMR source revision differs from the tested integration revision.")
        except (OSError, subprocess.SubprocessError):
            report["errors"].append("Could not verify GVHMR source revision.")
    for module in MODULES:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            report["missingModules"].append(module)
    for path, description in ASSETS:
        asset = repo / path
        size = asset.stat().st_size if asset.is_file() else 0
        report["assets"].append({"path": str(asset), "present": size > 1024 * 1024,
                                 "bytes": size, "description": description})
    if probe and not report["missingModules"] and not report["errors"]:
        try:
            import torch
            import torchvision
            from pytorch3d.transforms import axis_angle_to_matrix
            report["torch"] = torch.__version__
            report["hip"] = torch.version.hip
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable to PyTorch. CPU fallback is not supported by this integration.")
            device = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": device.name, "vramGB": round(device.total_memory / 1024**3, 1)}
            x = torch.ones((64, 64), device="cuda")
            assert torch.isfinite(x @ x.T).all().item()
            axis_angle_to_matrix(torch.zeros((2, 3), device="cuda"))
            torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device="cuda"), torch.ones(1, device="cuda"), .5)
            torch.cuda.synchronize()
            prepare_repo(repo)
            load_inference_modules()
            report["gpuAndImportsPassed"] = True
        except Exception as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
    report["prerequisitesPresent"] = not report["errors"] and not report["missingModules"] and all(a["present"] for a in report["assets"])
    report["readyForAttempt"] = report["prerequisitesPresent"] and bool(report.get("gpuAndImportsPassed"))
    return report
