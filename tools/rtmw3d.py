"""RTMW3D-X local capture; ungated Apache-2.0 models. See RTMW3D.md."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    commands.add_parser("setup")
    run = commands.add_parser("run")
    run.add_argument("video", type=Path)
    run.add_argument("output", type=Path, help="New raw .tracking.json, for the Node export step")
    run.add_argument("--fps", type=int, choices=(12, 24, 30), default=24)
    run.add_argument("--device", choices=("dml", "cpu"), default="dml")
    run.add_argument("--body-model", choices=("temporal", "legacy"), default="temporal",
                     help="Temporal MotionAGFormer body (default), or explicit legacy framewise RTMW3D")
    lift = commands.add_parser("lift", help="Run temporal body on saved native detections without re-decoding video")
    lift.add_argument("tracking", type=Path)
    lift.add_argument("output", type=Path)
    run.add_argument("--frames", type=int, help="Optional first N samples for an inference smoke test")
    run.add_argument("--height", type=float, default=1.7, help="Approximate actor height in meters; used only for estimated scale/depth")
    run.add_argument("--fov", type=float, default=60, help="Approximate horizontal camera FOV, degrees")
    args = parser.parse_args()
    if args.command == "setup":
        from backends.rtmw3d.install import setup
        setup(ROOT)
        return 0
    from backends.rtmw3d.models import download_models
    if args.command == "download":
        print(download_models(ROOT / ".models/rtmw3d"))
        from backends.rtmw3d.models import download_body2d
        print(download_body2d(ROOT / ".models/rtmw3d"))
        from backends.rtmw3d.temporal import download_model
        print(download_model(ROOT / ".models/rtmw3d"))
        return 0
    if args.command == "lift":
        import json
        from backends.rtmw3d.temporal import add_temporal_body
        if not args.tracking.is_file() or args.output.exists():
            parser.error("Tracking must exist; output must be a NEW file.")
        result = add_temporal_body(json.loads(args.tracking.read_text(encoding="utf8")), ROOT)
        from backends.rtmw3d.arm_refinement import refine_tracking_arms
        result = refine_tracking_arms(result)
        with args.output.open("x", encoding="utf8") as output:
            json.dump(result, output, allow_nan=False, separators=(",", ":"))
        return 0
    if not args.video.is_file() or args.output.exists():
        parser.error("Video must exist; output must be a NEW file. No originals are overwritten.")
    if not 1 <= args.height <= 2.5 or not 30 <= args.fov <= 100 or (args.frames is not None and not 1 <= args.frames <= 1800):
        parser.error("Height must be 1–2.5 m, FOV 30–100 degrees, sample limit 1–1,800.")
    from backends.rtmw3d.capture import capture_video
    capture_video(args, ROOT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"RTMW3D failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
