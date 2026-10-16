"""Reuse pinned rtmlib pre/postprocessing with explicit ORT provider settings.

No upstream files are modified. DML requires sequential execution and memory
patterns disabled. CPU is explicit; DML session failure is not hidden.
"""
from pathlib import Path
import sys
from .install import source_revision

REVISION = "03a1693e59e4f7cd84582c0fb30459b3bf18ad42"
Z_RANGE = 2.1744869


def decode_depth(simcc_z):
    # Official codec input_size is (288, 384, 288), not (288, 384, 384).
    # rtmlib's 2D model_input_size[-1] would incorrectly use image height.
    return (simcc_z / 144.0 - 1.0) * Z_RANGE


def load_models(root, pose_path, detector_path, device):
    import onnxruntime as ort

    repo = Path(root) / ".third-party/rtmlib"
    revision = source_revision(repo)
    if revision != REVISION:
        raise ValueError("Unexpected rtmlib revision; use the pinned setup script in a clean clone.")
    sys.path.insert(0, str(repo))
    from rtmlib import RTMPose3d, YOLOX

    provider = "DmlExecutionProvider" if device == "dml" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"{provider} unavailable. Install the isolated requirements or explicitly use --device cpu.")

    def initialize(tool, path, size, mean=None, std=None):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 4
        tool.session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
        if provider not in tool.session.get_providers():
            raise RuntimeError(f"{provider} session initialization failed; not silently falling back to CPU.")
        tool.session.disable_fallback()
        tool.model_input_size = size
        tool.mean, tool.std = mean, std
        tool.backend, tool.device = "onnxruntime", device
        print(f"Loaded {Path(path).name}: {tool.session.get_providers()}", flush=True)

    class Pose(RTMPose3d):
        def __init__(self):
            initialize(self, pose_path, (288, 384), (123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
            self.z_range, self.to_openpose = Z_RANGE, False

        def postprocess(self, outputs, center, scale, simcc_split_ratio=2.0):
            if tuple(array.shape[-1] for array in outputs) != (576, 768, 576):
                raise ValueError("Unexpected RTMW3D SimCC output dimensions.")
            keypoints, scores, simcc, image = super().postprocess(outputs, center, scale, simcc_split_ratio)
            keypoints[..., 2] = decode_depth(simcc[..., 2])
            return keypoints, scores, simcc, image

    class Detector(YOLOX):
        def __init__(self):
            initialize(self, detector_path, (640, 640))
            self.det_mode, self.nms_thr, self.score_thr = "human", .45, .5

    return Pose(), Detector()
