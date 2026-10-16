"""Optional headless GVHMR backend. Run doctor before run; see GVHMR Setup.md."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backends.gvhmr.diagnostics import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "run"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=ROOT / ".third-party/GVHMR")
        if name == "doctor":
            command.add_argument("--probe", action="store_true", help="Test GPU operators and headless inference imports; never load checkpoints.")
        else:
            command.add_argument("video", type=Path)
            command.add_argument("output", type=Path, help="New .gvhmr.json file; existing files are never overwritten")
            command.add_argument("--static-camera", action="store_true", help="Use only when the camera does not move")
            command.add_argument("--focal-mm", type=int, help="35mm-equivalent focal length, if known")
            command.add_argument("--accept-noncommercial", action="store_true", help="Confirm your use meets GVHMR's research/non-commercial license and you have obtained asset permissions")
            command.add_argument("--trust-checkpoints", action="store_true", help="Allow upstream pickle checkpoint loading. Only use official trusted weights; pickle can execute code")
    args = parser.parse_args()
    if args.command == "doctor":
        report = diagnose(args.repo, args.probe)
        print(json.dumps(report, indent=2))
        return 0 if (report["readyForAttempt"] if args.probe else report["prerequisitesPresent"]) else 2
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output. Choose a new filename.")
    if args.output.suffixes[-2:] != [".gvhmr", ".json"]:
        parser.error("Choose an output ending in .gvhmr.json (not MediaPipe .mocap.json).")
    if not args.accept_noncommercial or not args.trust_checkpoints:
        parser.error("Read GVHMR Setup.md. Running requires --accept-noncommercial and --trust-checkpoints; neither is assumed for you.")
    if args.focal_mm is not None and not 8 <= args.focal_mm <= 1000:
        parser.error("--focal-mm must be a 35mm-equivalent focal length between 8 and 1000 mm.")
    report = diagnose(args.repo, probe=True)
    if not report["readyForAttempt"]:
        print(json.dumps(report, indent=2))
        print("GVHMR is not ready. No inference was run and no capture output was created.", file=sys.stderr)
        return 2
    from backends.gvhmr.runner import run
    try:
        run(args, ROOT)
    except Exception as error:
        print(f"GVHMR failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
