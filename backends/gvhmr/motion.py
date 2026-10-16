"""Lossless body-joint transport to Blender; no MediaPipe landmark remapping.

Only built-ins are used so schema/conversion tests need no AI dependencies.
"""
import math

NAMES = ["Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee",
         "Spine2", "LeftAnkle", "RightAnkle", "Spine3", "LeftFoot", "RightFoot",
         "Neck", "LeftCollar", "RightCollar", "Head", "LeftShoulder", "RightShoulder",
         "LeftElbow", "RightElbow", "LeftWrist", "RightWrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def z_up(point):
    # Proper rotation (det=+1), not a reflection. Also valid for axis-angle
    # vectors: C R(a) C^-1 = R(Ca). GVHMR global coordinates have +Y up.
    return [float(point[0]), -float(point[2]), float(point[1])]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def build_capture(params, skeletons, metadata):
    """params are CPU lists from smpl_params_global; skeletons are unposed joints.

    Per-frame offsets preserve the model's shape estimates instead of silently
    fixing betas to the first frame. Root position includes the pelvis pivot,
    not just SMPL's transl parameter; rotation must NOT rotate that pivot twice.
    """
    count = len(skeletons)
    if not 1 <= count <= 1800:
        raise ValueError("GVHMR capture must contain 1–1,800 frames.")
    if any(len(params.get(key, [])) != count for key in ("body_pose", "global_orient", "transl")):
        raise ValueError("GVHMR parameter/frame counts do not match.")
    fps = metadata.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("Invalid GVHMR output frame rate.")
    frames = []
    for index, skeleton in enumerate(skeletons):
        if len(skeleton) != 22 or any(len(p) != 3 for p in skeleton):
            raise ValueError("Expected the 22 SMPL-X body joints.")
        pose, orient, transl = (params[key][index] for key in ("body_pose", "global_orient", "transl"))
        if len(pose) != 63 or len(orient) != 3 or len(transl) != 3:
            raise ValueError("Invalid GVHMR pose parameter dimensions.")
        offsets = [[0., 0., 0.]] + [z_up(subtract(skeleton[j], skeleton[PARENTS[j]])) for j in range(1, 22)]
        frames.append({"time": index / fps,
                       "root": z_up([skeleton[0][a] + transl[a] for a in range(3)]),
                       "rotations": [z_up(orient)] + [z_up(pose[j:j + 3]) for j in range(0, 63, 3)],
                       "offsets": offsets})
    origin = [frames[0]["root"][0], frames[0]["root"][1], 0.]
    for frame in frames:
        frame["root"] = subtract(frame["root"], origin)
        for vector in [frame["root"], *frame["rotations"], *frame["offsets"]]:
            if not all(math.isfinite(v) and abs(v) <= 10000 for v in vector):
                raise ValueError("GVHMR returned nonfinite or implausibly large parameters.")
    rest = [z_up(subtract(p, skeletons[0][0])) for p in skeletons[0]]
    return {"format": "framecraft-gvhmr", "version": 1,
            "coordinateSystem": "Blender Z-up meters; local axis-angle rotations in radians; root is world pelvis position",
            "metadata": {**metadata, "engine": "gvhmr", "layers": ["body", "root"],
                         "horizontalOrigin": origin, "duration": count / fps},
            "skeleton": {"names": NAMES.copy(), "parents": PARENTS.copy(), "restJoints": rest},
            "frames": frames}
