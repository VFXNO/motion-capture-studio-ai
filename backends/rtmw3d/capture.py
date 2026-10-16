"""Timestamp-sampled native 133-point inference, separate from Blender fitting."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .models import download_models
from .runtime import load_models, REVISION


def choose_box(boxes, previous=None):
    """Largest initial actor, then overlap association; never use an empty crop."""
    import numpy as np
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(axis=1) & ((boxes[:, 2:] - boxes[:, :2]) > 5).all(axis=1)]
    if not len(boxes):
        return None
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    if previous is None:
        return boxes[areas.argmax()]
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], previous[2:]) - np.maximum(boxes[:, :2], previous[:2])).prod(axis=1)
    overlap = intersection / np.maximum(1, areas + np.prod(previous[2:] - previous[:2]) - intersection)
    # Do not switch to another person if the original actor disappears.
    return boxes[overlap.argmax()] if overlap.max() >= .15 else None


def sample_video(path, fps, limit=None):
    """Nearest presentation-time frames, not frame-index/fps assumptions (VFR safe)."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else (
            container.duration / av.time_base if container.duration is not None else None)
        if duration is None or not 0 < duration <= 60:
            raise ValueError("Use a video with known duration of at most 60 seconds. Trim a separate copy first.")
        total = math.ceil(duration * fps - 1e-7)
        total = min(total, limit) if limit else total
        info = {"width": stream.width, "height": stream.height, "fps": fps,
                "sourceDuration": duration, "samples": total, "timing": "Nearest source presentation timestamp; constant output sample grid"}
        yield info
        decoded = iter(container.decode(video=0))
        before = next(decoded, None)
        if before is None or before.pts is None:
            raise ValueError("Video has no timestamped frames.")
        start = float(before.time)
        after = next(decoded, None)
        for index in range(total):
            target = index / fps
            while after is not None and float(after.time) - start < target:
                before, after = after, next(decoded, None)
            chosen = after if after is not None and abs(float(after.time) - start - target) < abs(float(before.time) - start - target) else before
            yield target, float(chosen.time) - start, chosen.to_ndarray(format="bgr24")


def capture_video(args, root):
    import numpy as np

    samples = sample_video(args.video, args.fps, args.frames)
    info = next(samples)
    pose_path, detector_path = download_models(root / ".models/rtmw3d")
    pose, detector = load_models(root, pose_path, detector_path, args.device)
    frames, previous = [], None
    started = time.monotonic()
    for index, (timestamp, decoded_time, image) in enumerate(samples):
        box = choose_box(detector(image), previous)
        record = {"time": timestamp, "sourceTime": decoded_time, "keypoints": None}
        if box is not None:
            previous = box
            keypoints, scores, _simcc, pixel = pose(image, [box])
            if keypoints.shape != (1, 133, 3) or not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
                raise ValueError("Invalid RTMW3D inference output.")
            record.update({"bbox": box.tolist(), "keypoints": np.column_stack((pixel[0], keypoints[0, :, 2], scores[0])).tolist()})
        frames.append(record)
        if index % 12 == 0 or index + 1 == info["samples"]:
            print(f"RTMW3D {index + 1}/{info['samples']}; {time.monotonic() - started:.1f}s; actor={'tracked' if box is not None else 'missing'}", flush=True)
    if not any(frame["keypoints"] for frame in frames):
        raise ValueError("No performer detected; no capture written.")
    data = {"format": "framecraft-rtmw3d-tracking", "version": 1,
            "metadata": {**info, "engine": "rtmw3d-x", "license": "Apache-2.0", "filename": args.video.name,
                         "device": args.device, "provider": pose.session.get_providers(), "sourceRevision": REVISION,
                         "heightMeters": args.height, "fov": args.fov, "createdAt": datetime.now(timezone.utc).isoformat(),
                         "coordinates": "Native COCO-WholeBody 133: [image_x_px, image_y_px, hip-relative_depth_m, score]. No global trajectory inferred."},
            "frames": frames}
    with Path(args.output).open("x", encoding="utf8") as handle:
        json.dump(data, handle, allow_nan=False, separators=(",", ":"))
    print(f"Saved native model predictions: {args.output}", flush=True)
