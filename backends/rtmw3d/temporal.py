"""MotionAGFormer-B temporal body inference; RTMW3D native predictions stay intact.

COCO -> H36M preprocessing follows TaatiTeam/MotionAGFormer demo/lib/preprocess.py
(Apache-2.0). Framecraft adds timestamp resampling, missing-body segmentation,
overlapping windows and flip augmentation. No pickle or PyTorch at runtime.
"""
import copy
import math
from pathlib import Path

from .models import download_one

MODEL = {
    "name": "motionagformer-b-243.onnx",
    "sha256": "8a5b3a384863ddc4c2279fe827dcf1b39a83609f867777725cc34e6d29697360",
    "url": "https://huggingface.co/jaybanez/MotionAGFormer-3d-ONNX/resolve/c1226b38f4f74e1ec966c169ec7de522a71ce34f/motionagformer_3d_243.onnx",
}
WINDOW, STRIDE, RATE = 243, 81, 30
COCO = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
H36M = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
FLIP = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def download_model(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return download_one(MODEL, directory)


def coco_to_h36m(keypoints, width, height):
    import numpy as np
    k = np.asarray(keypoints, dtype=np.float32)
    result = np.zeros((len(k), 17, 3), dtype=np.float32)
    result[:, H36M, :2] = k[:, COCO, :2]
    result[:, H36M, 2] = np.clip(k[:, COCO, 3], 0, 1)
    # Official demo's inferred head, thorax, pelvis and spine definitions.
    result[:, 10, 0] = k[:, 1:5, 0].mean(axis=1)
    result[:, 10, 1] = k[:, 1:3, 1].sum(axis=1) - k[:, 0, 1]
    shoulders = k[:, 5:7, :2].mean(axis=1)
    result[:, 8, :2] = shoulders + (k[:, 0, :2] - shoulders) / 3
    result[:, 0, :2] = k[:, 11:13, :2].mean(axis=1)
    result[:, 7, :2] = k[:, [5, 6, 11, 12], :2].mean(axis=1)
    result[:, 9, :2] -= (result[:, 9, :2] - shoulders) / 4
    result[:, 7, 0] += 2 * (result[:, 7, 0] - result[:, [0, 8], 0].mean(axis=1))
    result[:, 8, 1] -= (k[:, 1:3, 1].mean(axis=1) - k[:, 0, 1]) * 2 / 3
    result[:, 0, 2] = k[:, 11:13, 3].mean(axis=1)
    result[:, 8, 2] = k[:, 5:7, 3].mean(axis=1)
    result[:, 7, 2] = result[:, [0, 8], 2].mean(axis=1)
    result[:, 10, 2] = k[:, 1:5, 3].mean(axis=1)
    # Letterbox portrait inputs to a square before the official screen-space
    # normalization. Do not stretch tall videos into the landscape training
    # range or let a 9:16 dancer occupy nearly four normalized Y units.
    result[..., :2] = (result[..., :2] - np.array([width / 2, height / 2], dtype=np.float32)) * (2 / max(width, height))
    result[..., 2] = np.clip(result[..., 2], 0, 1)
    return result


def flip_pose(value):
    result = value[..., FLIP, :].copy()
    result[..., 0] *= -1
    return result


def segments(frames):
    current = []
    for index, frame in enumerate(frames):
        k = body_keypoints(frame)
        valid = k is not None and min(k[j][3] for j in (5, 6, 11, 12)) >= .35
        if not valid or (current and frame["time"] - frames[current[-1]]["time"] > .15):
            if current:
                yield current
            current = []
        if valid:
            current.append(index)
    if current:
        yield current


def body_keypoints(frame):
    """Use the dedicated body detector when present; native wholebody stays raw."""
    import numpy as np
    if frame.get("body2d") is not None:
        body = np.asarray(frame["body2d"], dtype=np.float32)
        if body.shape != (17, 3) or not np.isfinite(body).all() or ((body[:, 2] < 0) | (body[:, 2] > 1)).any():
            raise ValueError("Invalid dedicated body image predictions.")
        return np.column_stack((body[:, :2], np.zeros(17), body[:, 2]))
    return frame.get("keypoints")


def infer_sequence(values, times, predict):
    """Use real seconds, overlapping contexts and positive blend weights at ends."""
    import numpy as np
    times = np.asarray(times, dtype=float)
    grid = times[0] + np.arange(max(1, math.ceil((times[-1] - times[0]) * RATE) + 1)) / RATE
    values = np.asarray(values, dtype=np.float32)
    uniform = np.empty((len(grid), 17, 3), dtype=np.float32)
    for joint in range(17):
        reliable = values[:, joint, 2] >= .35
        for axis in range(3):
            # Fill unreliable XY only for network context; observed confidence
            # is retained, and original detections are never rewritten.
            mask = reliable if axis < 2 and reliable.any() else np.ones(len(times), dtype=bool)
            uniform[:, joint, axis] = np.interp(grid, times[mask], values[mask, joint, axis])
    total = np.zeros((len(grid), 17, 3), dtype=np.float64)
    weight = np.zeros(len(grid), dtype=np.float64)
    half = WINDOW // 2
    for center in range(0, len(grid) + STRIDE // 2, STRIDE):
        indices = np.arange(center - half, center + half + 1)
        window = uniform[np.clip(indices, 0, len(grid) - 1)][None]
        normal = predict(window)
        mirrored = flip_pose(predict(flip_pose(window)))
        prediction = (normal + mirrored)[0] * .5
        if prediction.shape != (WINDOW, 17, 3) or not np.isfinite(prediction).all():
            raise ValueError("Invalid temporal body inference output.")
        prediction -= prediction[0:len(prediction), 0:1, :].copy()
        valid = (indices >= 0) & (indices < len(grid))
        weights = np.maximum(.02, np.hanning(WINDOW))[valid]
        total[indices[valid]] += prediction[valid] * weights[:, None, None]
        weight[indices[valid]] += weights
    if not (weight > 0).all():
        raise ValueError("Temporal window coverage failed.")
    total /= weight[:, None, None]
    result = np.empty((len(times), 17, 3), dtype=np.float32)
    for joint in range(17):
        for axis in range(3):
            result[:, joint, axis] = np.interp(times, grid, total[:, joint, axis])
    return result


def add_temporal_body(tracking, root, predict=None):
    import numpy as np
    if tracking.get("format") != "framecraft-rtmw3d-tracking" or tracking.get("version") != 1:
        raise ValueError("Expected native RTMW3D tracking for temporal body inference.")
    frames, metadata = tracking["frames"], tracking["metadata"]
    if not 1 <= len(frames) <= 1800 or not all(isinstance(f.get("time"), (int,float)) and math.isfinite(f["time"]) and f["time"] >= 0 for f in frames):
        raise ValueError("Invalid temporal sample count/times.")
    if any(not isinstance(metadata.get(key), (int,float)) or not math.isfinite(metadata[key]) or not 0 < metadata[key] <= 32768 for key in ("width","height")):
        raise ValueError("Invalid temporal input dimensions.")
    if any(b["time"] <= a["time"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Temporal timestamps must increase.")
    if frames[-1]["time"] - frames[0]["time"] > 60:
        raise ValueError("Temporal clips must be at most 60 seconds.")
    for frame in frames:
        if frame.get("keypoints") is not None:
            k = np.asarray(frame["keypoints"])
            if k.shape != (133, 4) or not np.isfinite(k).all():
                raise ValueError("Invalid native keypoints.")
    if predict is None:
        import onnxruntime as ort
        model = download_model(Path(root) / ".models/rtmw3d")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        # Deliberate CPU stage, not a silent fallback. The graph's top-k GCN
        # fails DML initialization; this small sequence model is fast on CPU.
        session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        session.disable_fallback()
        if session.get_inputs()[0].shape != [1, WINDOW, 17, 3]:
            raise ValueError("Unexpected temporal model shape.")
        predict = lambda x: session.run(None, {session.get_inputs()[0].name: x.astype(np.float32)})[0]
    result = copy.deepcopy(tracking)
    for frame in result["frames"]:
        frame.pop("temporalBody", None)
    for part in segments(frames):
        inputs = coco_to_h36m([body_keypoints(frames[i]) for i in part], metadata["width"], metadata["height"])
        poses = infer_sequence(inputs, [frames[i]["time"] for i in part], predict)
        for index, pose in zip(part, poses):
            result["frames"][index]["temporalBody"] = pose.tolist()
        print(f"Temporal body {part[-1] + 1}/{len(frames)}; MotionAGFormer-B on CPU", flush=True)
    if not any(f.get("temporalBody") for f in result["frames"]):
        raise ValueError("No reliable body sequence for temporal inference.")
    result["metadata"]["bodyEstimator"] = {
        "model": "motionagformer-b", "window": WINDOW, "stride": STRIDE, "inferenceFps": RATE,
        "provider": "CPUExecutionProvider", "sha256": MODEL["sha256"],
        "layout": "h36m17", "coordinates": "Camera X right, Y down, Z away; root-relative model units",
        "normalization": "centered-square-letterbox",
        "method": "overlap-and-flip-v1", "source": "https://github.com/TaatiTeam/MotionAGFormer",
        "bodyDetector": metadata.get("bodyDetector", "rtmw3d-x"),
        "assumptions": "2D-to-3D temporal body model with scaled-orthographic projection. Depth remains inferred; scale calibrated once to actor height. Hands/face remain RTMW3D.",
    }
    return result
