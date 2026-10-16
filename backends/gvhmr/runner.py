"""Headless adaptation of GVHMR tools/demo/demo.py; see adjacent LICENSE.

No compiled mesh renderer/DPVO, no cloud calls, no automatic model downloads.
All upstream relative checkpoint paths are resolved inside the chosen repo.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import gc
import json
import os
from pathlib import Path
import subprocess
import uuid

from .diagnostics import REVISION, prepare_repo, load_inference_modules
from .motion import build_capture, PARENTS


@contextmanager
def trusted_checkpoint_loading():
    # Only entered after the CLI's explicit --trust-checkpoints acknowledgement.
    # Upstream checkpoints contain pickle metadata unsupported by the new safe
    # default. Scope this compatibility override to this process/run and restore.
    import torch
    original = torch.load

    @functools.wraps(original)
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def normalize_video(source, destination):
    import av
    import imageio_ffmpeg
    # Read source timing, not just frame count: GVHMR assumes 30Hz internally.
    # The upstream demo writes at 30Hz without resampling; that changes speed for
    # other frame rates. Here FFmpeg really resamples timestamps to 30Hz.
    with av.open(str(source)) as container:
        video = container.streams.video[0]
        duration = float(video.duration * video.time_base) if video.duration is not None else (
            container.duration / av.time_base if container.duration is not None else None)
    if duration is None or not 0 < duration <= 60:
        raise ValueError("Use a video with known duration of at most 60 seconds (1,800 output frames). Trim a separate copy first.")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n",
                    "-i", str(source), "-map", "0:v:0", "-an", "-vf", "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(destination)],
                   check=True, timeout=300)
    with av.open(str(destination)) as container:
        stream = container.streams.video[0]
        count = sum(1 for _ in container.decode(video=0))
        width, height = stream.width, stream.height
    if not 2 <= count <= 1800:
        raise ValueError("Normalized video must contain 2–1,800 frames.")
    return {"fps": 30, "width": width, "height": height, "frames": count,
            "sourceDuration": duration, "timing": "Resampled to constant 30fps using source timestamps"}


def release_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def predict(video, info, static_camera=False, focal_mm=None):
    import torch
    import hydra
    from hmr4d.utils.preproc import Tracker, VitPoseExtractor, Extractor, SimpleVO
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu

    cfg = load_inference_modules()
    if focal_mm is not None:
        cfg.f_mm = focal_mm
    with torch.inference_mode():
        print("GVHMR: detecting/tracking one performer", flush=True)
        detector = Tracker()
        boxes = get_bbx_xys_from_xyxy(detector.get_one_track(str(video)).float(), base_enlarge=1.2).float()
        del detector
        release_gpu()
        print("GVHMR: ViTPose huge 2D pose estimation", flush=True)
        pose_model = VitPoseExtractor()
        keypoints = pose_model.extract(str(video), boxes)
        del pose_model
        release_gpu()
        print("GVHMR: HMR2.0a visual features", flush=True)
        feature_model = Extractor()
        features = feature_model.extract_video_features(str(video), boxes)
        del feature_model
        release_gpu()
        count, width, height = info["frames"], info["width"], info["height"]
        if static_camera:
            camera_rotations = torch.eye(3).repeat(count, 1, 1)
        else:
            print("GVHMR: estimating camera rotation with SimpleVO", flush=True)
            trajectory = SimpleVO(str(video), scale=.5, step=8, method="sift", f_mm=focal_mm).compute()
            camera_rotations = torch.from_numpy(trajectory[:, :3, :3]).float()
        intrinsics = create_camera_sensor(width, height, focal_mm)[2] if focal_mm else estimate_K(width, height)
        data = {"length": torch.tensor(count), "bbx_xys": boxes, "kp2d": keypoints,
                "K_fullimg": intrinsics.repeat(count, 1, 1), "cam_angvel": compute_cam_angvel(camera_rotations),
                "f_imgseq": features}
        if any(len(data[key]) != count for key in ("bbx_xys", "kp2d", "cam_angvel", "f_imgseq")):
            raise ValueError("Preprocessing changed the frame count; refusing a mistimed export.")
        print("GVHMR: temporal body + world trajectory + contact postprocessing", flush=True)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        prediction = model.predict(data, static_cam=static_camera)
        params = prediction["smpl_params_global"]
        body_model = model.pipeline.endecoder.smplx_model
        if body_model.parents[:22].tolist() != PARENTS:
            raise ValueError("Unsupported SMPL-X body hierarchy.")
        skeletons = body_model.get_skeleton(params["betas"])[..., :22, :].cpu().tolist()
        prediction = detach_to_cpu(prediction)
        params = {key: value.cpu().tolist() for key, value in params.items()}
        return prediction, params, skeletons


def run(args, root):
    import torch
    from blender_importer import validate_capture

    repo = prepare_repo(args.repo)
    source, output = args.video.resolve(), args.output.resolve()
    # Every attempt has a new working folder. No stale cache from a different
    # video, lens setting, or camera mode can be silently reused.
    work = root / ".gvhmr-runs" / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    print(f"GVHMR working directory: {work}", flush=True)
    normalized = work / "input-30fps.mp4"
    info = normalize_video(source, normalized)
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        with trusted_checkpoint_loading():
            prediction, params, skeletons = predict(normalized, info, args.static_camera, args.focal_mm)
    finally:
        os.chdir(old_cwd)
    torch.save(prediction, work / "hmr4d_results.pt")
    metadata = {**info, "filename": source.name, "sourceRevision": REVISION,
                "staticCamera": args.static_camera, "focalMm": args.focal_mm,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "notes": "GVHMR body/root only; fingers and expressions are not inferred. No additional MediaPipe smoothing or root fit."}
    capture = build_capture(params, skeletons, metadata)
    validate_capture(capture)  # Same validation Blender uses, before output creation.
    encoded = json.dumps(capture, allow_nan=False, separators=(",", ":"))
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(output), "frames": len(capture["frames"]), "engine": "GVHMR",
                      "rawPrediction": str(work / "hmr4d_results.pt"), "layers": ["body", "root"]}, indent=2))
