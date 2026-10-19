"""Framecraft capture importer for Blender 4.2+.

Open this file in Blender's Text Editor and Run Script, then use
File > Import > Framecraft Capture (.mocap.json / .gvhmr.json). No external packages needed.
"""
import json
import math
import re
import statistics
from pathlib import Path

bl_info = {
    "name": "Framecraft Motion Capture", "author": "Framecraft",
    "version": (0, 7, 1), "blender": (4, 2, 0),
    "location": "File > Import", "category": "Import-Export",
    "description": "Import body/finger animation, facial wire mesh, and expression scores",
}

try:
    import bpy
    from bpy.props import BoolProperty, IntProperty, StringProperty
    from bpy_extras.io_utils import ImportHelper
    from mathutils import Vector, Matrix, Quaternion
except ImportError:
    bpy = None


def validate_capture(data):
    """Validate before changing the scene; this importer never executes capture data."""
    if isinstance(data, dict) and data.get("format") == "framecraft-gvhmr":
        return validate_gvhmr_capture(data)
    if not isinstance(data, dict) or data.get("format") != "framecraft-mocap" or data.get("version") != 1:
        raise ValueError("Expected a Framecraft version 1 capture (.mocap.json).")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid capture metadata.")
    face68 = data.get("faceTopology") == "coco-wholebody-68"
    if data.get("faceTopology") not in (None, "coco-wholebody-68") or (face68 and metadata.get("engine") != "rtmw3d-x"):
        raise ValueError("Unsupported facial topology/engine.")
    face_count = 68 if face68 else 478
    image_fit = metadata.get("poseFitting") is not None
    model_body = metadata.get("bodyEstimator") is not None
    if model_body and (not isinstance(metadata["bodyEstimator"], dict) or metadata["bodyEstimator"].get("model") != "motionagformer-b"
                       or metadata["bodyEstimator"].get("rigFitting") != "model-directions-v1" or image_fit):
        raise ValueError("Unsupported temporal body fitting settings.")
    if image_fit:
        calibration = metadata.get("calibration")
        camera = calibration.get("camera") if isinstance(calibration, dict) else None
        if metadata.get("engine") != "rtmw3d-x" or not isinstance(metadata["poseFitting"], dict) or metadata["poseFitting"].get("method") != "image-constrained-limbs-v1":
            raise ValueError("Unsupported image-constrained fitting settings.")
        if not isinstance(camera, dict) or any(not isinstance(camera.get(key), (int, float)) or not math.isfinite(camera[key]) or not .01 < camera[key] < 100000
               for key in ("focal", "width", "height")):
            raise ValueError("Invalid projection camera calibration.")
        if any(camera[key] != metadata.get(key) for key in ("width", "height")):
            raise ValueError("Projection camera dimensions must match the source video.")
    fps = metadata.get("fps", 0)
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("Capture frame rate must be between 0 and 240.")
    for key in ("width", "height"):
        value = metadata.get(key, 0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 32768:
            raise ValueError("Invalid video dimensions.")
    frames = data.get("frames")
    if not isinstance(frames, list) or not 0 < len(frames) <= 1800:
        raise ValueError("Capture must contain between 1 and 1,800 frames.")
    previous_time = -1
    root_config = data.get("rootMotion")
    if root_config is not None:
        if not isinstance(root_config, dict) or not isinstance(root_config.get("enabled"), bool):
            raise ValueError("Invalid root motion settings.")
        if root_config["enabled"]:
            pitch = root_config.get("pitch")
            if root_config.get("method") != "grounded-perspective-v1" or not isinstance(pitch, (int, float)) or not math.isfinite(pitch) or not -15 <= pitch <= 60:
                raise ValueError("Unsupported root motion coordinate settings.")

    def check_points(points, count):
        if not isinstance(points, list) or len(points) != count:
            raise ValueError(f"Expected {count} landmarks.")
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("Invalid landmark.")
            for axis in ("x", "y", "z"):
                value = point.get(axis)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e6:
                    raise ValueError("Landmark coordinates must be finite numbers.")

    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid capture frame.")
        time = frame.get("time")
        if not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0 or time <= previous_time:
            raise ValueError("Frame times must be finite, nonnegative, and strictly increasing.")
        previous_time = time
        root = frame.get("root")
        if root is not None:
            if not isinstance(root, dict) or root.get("status") not in ("tracked", "held"):
                raise ValueError("Invalid root sample.")
            position = root.get("position", {})
            if not isinstance(position, dict):
                raise ValueError("Invalid root position.")
            for axis in ("x", "y", "z"):
                value = position.get(axis)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10000:
                    raise ValueError("Root positions must be finite estimated-meter coordinates.")
        if frame.get("body"):
            check_points(frame["body"].get("world"), 33)
            if model_body:
                check_points(frame["body"].get("pose17"), 17)
            if image_fit:
                check_points(frame["body"].get("landmarks"), 33)
                for point in frame["body"]["landmarks"]:
                    score = point.get("visibility", 1)
                    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid image-joint confidence.")
        if (image_fit and frame.get("body")) or ((image_fit or model_body) and frame.get("projection") is not None):
            projection = frame.get("projection")
            if not isinstance(projection, dict):
                raise ValueError("Image-constrained body samples need a projection center.")
            check_points([projection.get("cameraCenter")], 1)
            if projection["cameraCenter"]["z"] <= .1:
                raise ValueError("Projection center must be in front of the camera.")
            if model_body:
                scale = projection.get("pixelsPerMeter")
                if projection.get("model") != "scaled-orthographic" or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 10 <= scale <= 1e6:
                    raise ValueError("Invalid temporal camera scale.")
        hands = frame.get("hands", [])
        if not isinstance(hands, list) or len(hands) > 2:
            raise ValueError("Expected no more than two hands per frame.")
        for hand in hands:
            if hand.get("side") not in ("Left", "Right"):
                raise ValueError("Unknown hand side.")
            check_points(hand.get("world"), 21)
        if frame.get("face"):
            check_points(frame["face"].get("landmarks"), face_count)
            if face68:
                if frame["face"].get("topology") != "coco-wholebody-68" or frame["face"].get("blendshapes"):
                    raise ValueError("Native RTMW3D face68 has no expression scores.")
                check_points(frame["face"].get("world"), 68)
            for name, value in frame["face"].get("blendshapes", {}).items():
                if not re.fullmatch(r"[a-zA-Z0-9_]{1,64}", name) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Invalid expression score.")
    edges = data.get("faceEdges", [])
    if not isinstance(edges, list) or len(edges) > 5000:
        raise ValueError("Invalid face topology.")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(i, int) and 0 <= i < face_count for i in edge):
            raise ValueError("Invalid face edge.")
    if root_config and root_config["enabled"] and not any(f.get("root", {}).get("status") == "tracked" for f in frames if f.get("root")):
        raise ValueError("Enabled root motion needs at least one tracked root sample.")
    return data


GVHMR_NAMES = ["Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee",
               "Spine2", "LeftAnkle", "RightAnkle", "Spine3", "LeftFoot", "RightFoot",
               "Neck", "LeftCollar", "RightCollar", "Head", "LeftShoulder", "RightShoulder",
               "LeftElbow", "RightElbow", "LeftWrist", "RightWrist"]
GVHMR_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def validate_gvhmr_capture(data):
    """Separate, explicit rotation schema: never reinterpret it as MediaPipe."""
    if data.get("version") != 1:
        raise ValueError("Expected a Framecraft GVHMR version 1 capture (.gvhmr.json).")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("engine") != "gvhmr":
        raise ValueError("Missing GVHMR engine metadata.")

    def number(value, bound):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and abs(value) <= bound

    fps = metadata.get("fps")
    if not number(fps, 240) or fps <= 0:
        raise ValueError("Invalid GVHMR frame rate.")
    if any(not number(metadata.get(key), 32768) or metadata[key] <= 0 for key in ("width", "height")):
        raise ValueError("Invalid GVHMR video dimensions.")
    skeleton = data.get("skeleton")
    if not isinstance(skeleton, dict) or skeleton.get("names") != GVHMR_NAMES or skeleton.get("parents") != GVHMR_PARENTS:
        raise ValueError("Unsupported GVHMR body hierarchy.")

    def vector(value, bound=10000):
        if not isinstance(value, list) or len(value) != 3 or not all(number(v, bound) for v in value):
            raise ValueError("GVHMR vectors must contain three finite numbers.")

    def vectors(values):
        if not isinstance(values, list) or len(values) != 22:
            raise ValueError("Expected 22 GVHMR body-joint vectors.")
        for value in values:
            vector(value)

    vectors(skeleton.get("restJoints"))
    if any(abs(v) > 1e-5 for v in skeleton["restJoints"][0]):
        raise ValueError("GVHMR rest pelvis must be centered; translation belongs on the root.")
    frames = data.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 1800:
        raise ValueError("GVHMR capture must contain 1–1,800 frames.")
    previous = -1
    for frame in frames:
        if not isinstance(frame, dict) or not number(frame.get("time"), 86400) or frame["time"] < 0 or frame["time"] <= previous:
            raise ValueError("GVHMR frame times must be nonnegative and strictly increasing.")
        previous = frame["time"]
        vector(frame.get("root"))
        vectors(frame.get("rotations"))
        vectors(frame.get("offsets"))
        if any(abs(v) > 1e-5 for v in frame["offsets"][0]):
            raise ValueError("GVHMR pelvis offset must be zero; translation belongs on the root.")
    return data


BODY_BONES = [
    ("Hips", "hips", "spine", None), ("Spine", "spine", "chest", "Hips"),
    ("Chest", "chest", "neck", "Spine"), ("Neck", "neck", "head", "Chest"),
    ("Head", "head", "head_top", "Neck"),
]
for _side in ("Left", "Right"):
    BODY_BONES.extend([
        (f"{_side}Shoulder", "chest", f"{_side}shoulder", "Chest"),
        (f"{_side}UpperArm", f"{_side}shoulder", f"{_side}elbow", f"{_side}Shoulder"),
        (f"{_side}LowerArm", f"{_side}elbow", f"{_side}wrist", f"{_side}UpperArm"),
        (f"{_side}Hand", f"{_side}wrist", f"{_side}palm", f"{_side}LowerArm"),
        (f"{_side}UpperLeg", f"{_side}hip", f"{_side}knee", "Hips"),
        (f"{_side}LowerLeg", f"{_side}knee", f"{_side}ankle", f"{_side}UpperLeg"),
        (f"{_side}Foot", f"{_side}ankle", f"{_side}toe", f"{_side}LowerLeg"),
    ])


def ground_coordinates(point, pitch=0):
    """Same camera-to-ground transform as root-motion.mjs, testable without Blender."""
    angle = math.radians(pitch)
    return (point["x"], math.cos(angle) * point["z"] - math.sin(angle) * point["y"],
            -math.sin(angle) * point["z"] - math.cos(angle) * point["y"])


def converted(point, pitch=0):
    return Vector(ground_coordinates(point, pitch))


def root_location(frame, fallback):
    root = frame.get("root")
    return tuple(root["position"][axis] for axis in ("x", "y", "z")) if root else fallback


def root_motion_summary(data):
    if data.get("format") == "framecraft-gvhmr":
        return {axis: max(f["root"][i] for f in data["frames"]) - min(f["root"][i] for f in data["frames"])
                for i, axis in enumerate(("x", "y", "z"))}
    if not (data.get("rootMotion") or {}).get("enabled"):
        return None
    positions = [frame["root"]["position"] for frame in data["frames"]
                 if frame.get("root") and frame["root"].get("status") == "tracked"]
    if not positions:
        return None
    return {axis: max(point[axis] for point in positions) - min(point[axis] for point in positions)
            for axis in ("x", "y", "z")}


def body_positions(body, offset, pitch=0, centered=False):
    if not body:
        return {}
    center = {axis: (body["world"][23][axis] + body["world"][24][axis]) / 2 if centered else 0
              for axis in ("x", "y", "z")}
    p = [converted({axis: point[axis] - center[axis] for axis in ("x", "y", "z")}, pitch) + offset
         for point in body["world"]]
    hips = (p[23] + p[24]) / 2
    chest = (p[11] + p[12]) / 2
    head = (p[7] + p[8]) / 2
    up = head - chest
    if up.length < 0.001:
        up = Vector((0, 0, 1))
    result = {"hips": hips, "spine": hips.lerp(chest, 0.45), "chest": chest,
              "neck": chest.lerp(head, 0.65), "head": head,
              "head_top": head + up.normalized() * 0.12, "nose": p[0]}
    if body.get("pose17"):
        native = [converted({axis: point[axis] - center[axis] for axis in ("x", "y", "z")}, pitch) + offset for point in body["pose17"]]
        result.update(hips=hips, spine=native[7], chest=native[8], neck=native[9],
                      head=native[9].lerp(native[10], .5), head_top=native[10])
    for side, indices in (("Left", (11, 13, 15, 19, 23, 25, 27, 31)),
                          ("Right", (12, 14, 16, 20, 24, 26, 28, 32))):
        for name, index in zip(("shoulder", "elbow", "wrist", "palm", "hip", "knee", "ankle", "toe"), indices):
            result[side + name] = p[index]
    result["Leftheel"], result["Rightheel"] = p[29], p[30]
    for side, index in (("Left", 19), ("Right", 20)):
        wrist, palm = result[side + "wrist"], result[side + "palm"]
        landmarks = body.get("landmarks", body["world"])
        if landmarks[index].get("visibility", 1) < .35 or not .02 <= (palm - wrist).length <= .16:
            # Body33's index fingertip may be an invalid wholebody hand depth.
            # Use a small display-only hand bone, never that untrusted distance.
            forearm = wrist - result[side + "elbow"]
            direction = forearm.normalized() if forearm.length > .001 else Vector((0, 0, -1))
            result[side + "palm"] = wrist + direction * min(.1, max(.04, forearm.length * .28))
    return result


def segment_lengths(frames):
    """Use robust clip-wide lengths rather than one potentially occluded frame."""
    result = {}
    for side, indices in (("Left", (11, 13, 15, 23, 25, 27, 31)),
                          ("Right", (12, 14, 16, 24, 26, 28, 32))):
        for label, a, b in (("arm", indices[0], indices[1]), ("forearm", indices[1], indices[2]),
                            ("thigh", indices[3], indices[4]), ("shin", indices[4], indices[5]),
                            ("foot", indices[5], indices[6])):
            values = []
            for frame in frames:
                body = frame.get("body")
                if not body:
                    continue
                landmarks = body.get("landmarks", body["world"])
                if min(landmarks[a].get("visibility", 1), landmarks[b].get("visibility", 1)) < .5:
                    continue
                length = math.dist(tuple(body["world"][a][axis] for axis in ("x", "y", "z")),
                                   tuple(body["world"][b][axis] for axis in ("x", "y", "z")))
                if .025 < length < 1.2:
                    values.append(length)
            if values:
                result[side + label] = statistics.median(values)
    for label, a, b, minimum, maximum in (("shoulder_width", 11, 12, .12, .7), ("hip_width", 23, 24, .08, .5)):
        values = [math.dist(tuple(f["body"]["world"][a][axis] for axis in ("x", "y", "z")),
                            tuple(f["body"]["world"][b][axis] for axis in ("x", "y", "z")))
                  for f in frames if f.get("body") and
                  min(f["body"]["world"][i].get("visibility", 1) for i in (a, b)) >= .5]
        values = [value for value in values if minimum < value < maximum]
        if values:
            result[label] = statistics.median(values)
    return result


FINGER_CHAINS = (("Thumb", 1), ("Index", 5), ("Middle", 9), ("Ring", 13), ("Pinky", 17))
HAND_EDGES = tuple(edge for _, start in FINGER_CHAINS
                   for edge in ((0, start), (start, start + 1), (start + 1, start + 2), (start + 2, start + 3)))


def hand_measurements(hand, scale=1):
    """Geometry guard independent of Blender. Confidence alone is insufficient."""
    points = hand["world"]
    if hand.get("score", 1) < .35 or any(p.get("visibility", 1) < .35 for p in points):
        return None
    xyz = [tuple(p[axis] for axis in ("x", "y", "z")) for p in points]
    lengths = {edge: math.dist(xyz[edge[0]], xyz[edge[1]]) for edge in HAND_EDGES}
    if any(not (.005 * scale <= length <= .16 * scale if a == 0 else .002 * scale <= length <= .085 * scale)
           for (a, _), length in lengths.items()):
        return None
    if any(math.dist(xyz[0], point) > .28 * scale for point in xyz):
        return None
    return lengths


def hand_profiles(frames, metadata):
    """Robust clip-wide hand sizes and a representative *valid* rest sample.

    Also protects older .mocap.json files exported before the native quality gate.
    No fallback finger detections/anthropometric fingers are invented.
    """
    scale = metadata.get("heightMeters", 1.7) / 1.7
    profiles = {}
    for side in ("Left", "Right"):
        candidates = [(frame, hand, measured) for frame in frames for hand in frame.get("hands", [])
                      if hand["side"] == side and (measured := hand_measurements(hand, scale)) is not None]
        if not candidates:
            continue
        lengths = {edge: statistics.median(measured[edge] for _, _, measured in candidates) for edge in HAND_EDGES}
        candidates = [item for item in candidates if all(.4 <= item[2][edge] / length <= 2.5
                                                       for edge, length in lengths.items())]
        if not candidates:
            continue
        def cost(candidate):
            _, hand, measured = candidate
            return sum(abs(math.log(measured[edge] / lengths[edge])) for edge in HAND_EDGES) - hand.get("score", 1)
        first, _, _ = min(candidates, key=cost)
        profiles[side] = {"lengths": lengths, "frame": first, "scale": scale}
    return profiles


def usable_hand(hand, profile):
    measured = hand_measurements(hand, profile["scale"])
    return measured is not None and all(.4 <= measured[edge] / length <= 2.5
                                       for edge, length in profile["lengths"].items())


def fit_hand_positions(points, lengths):
    """Keep observed directions, but use fixed lengths and connected chains."""
    result = {0: points[0]}
    for _, start in FINGER_CHAINS:
        for a, b in ((0, start), (start, start + 1), (start + 1, start + 2), (start + 2, start + 3)):
            result[b] = result[a] + (points[b] - points[a]).normalized() * lengths[(a, b)]
    return [result[i] for i in range(21)]


def smooth_point_track(values, times, confidence, speed_limit, sigma):
    """Centered, time-based cleanup. A noisy high speed must not open the filter.

    Forward/backward speed projections are averaged to avoid a one-way lag.
    Callers split detection/time gaps; this never fills absent body samples.
    """
    def limit(reverse):
        output = list(values)
        indices = list(range(len(values)))
        if reverse:
            indices.reverse()
        for previous, index in zip(indices, indices[1:]):
            distance = math.dist(output[previous], values[index])
            maximum = speed_limit * abs(times[index] - times[previous])
            alpha = min(1, maximum / max(distance, 1e-9))
            output[index] = tuple(a + (b - a) * alpha for a, b in zip(output[previous], values[index]))
        return output
    forward, backward = limit(False), limit(True)
    limited = [tuple((a + b) / 2 for a, b in zip(f, b)) for f, b in zip(forward, backward)]
    result = []
    for index, time in enumerate(times):
        neighbors = range(max(0, index - 24), min(len(times), index + 25))
        weights = [(j, math.exp(-.5 * ((times[j] - time) / sigma) ** 2) * max(.05, confidence[j]))
                   for j in neighbors if abs(times[j] - time) <= sigma * 3]
        total = sum(weight for _, weight in weights)
        result.append(tuple(sum(limited[j][axis] * weight for j, weight in weights) / total for axis in range(3)))
    return result


def smooth_body_frames(frames, strength, scale=1):
    """Smooth RTMW3D body targets *before* fixed-length IK, not baked bone heads.

    Root trajectory, contacts, facial data, missing samples, and source data are
    left unchanged. Foot locking/floor constraints still run after this step.
    """
    if strength <= 0:
        return frames
    result = list(frames)
    segments, current = [], []
    for index, frame in enumerate(frames):
        if not frame.get("body") or (current and frame["time"] - frames[current[-1]]["time"] > .15):
            if current:
                segments.append(current)
            current = []
        if frame.get("body"):
            current.append(index)
    if current:
        segments.append(current)
    for segment in segments:
        times = [frames[i]["time"] for i in segment]
        worlds = [[dict(p) for p in frames[i]["body"]["world"]] for i in segment]
        for joint in range(33):
            values = [tuple(frames[i]["body"]["world"][joint][a] for a in ("x", "y", "z")) for i in segment]
            confidence = [frames[i]["body"]["world"][joint].get("visibility", 1) for i in segment]
            # Generous dance-speed bounds suppress single-frame depth teleports,
            # not ordinary fast arms/feet. They scale with performer height.
            speed = 8 if joint in (15, 16, 17, 18, 19, 20, 21, 22) else 7 if joint >= 27 else 6
            cleaned = smooth_point_track(values, times, confidence, speed * scale, .022 + .046 * strength)
            for points, value in zip(worlds, cleaned):
                points[joint].update(zip(("x", "y", "z"), value))
        for index, world in zip(segment, worlds):
            result[index] = {**frames[index], "body": {**frames[index]["body"], "world": world}}
    return result


def transport_orientation(direction, rest, previous=None):
    """Aim without inventing a new axial twist from the rest pose every frame."""
    rotation = previous if previous is not None else rest
    return (rotation @ Vector((0, 1, 0))).rotation_difference(direction.normalized()) @ rotation


def smooth_hand_animation(writer, specs, times, numbers, tracked, strength):
    """Quaternion-safe cleanup and bounded re-acquisition in parent space.

    Filtering local transforms keeps phalanges connected and at fixed length.
    Missing detections hold the filtered pose; only re-acquisition is blended.
    """
    if strength <= 0:
        return
    sigma = .022 + .045 * strength
    for name, *_ in specs:
        side = "Left" if name.startswith("Left") else "Right" if name.startswith("Right") else None
        if side not in tracked or not (name == side + "Hand" or any(finger in name for finger, _ in FINGER_CHAINS)):
            continue
        rotation_path = f'pose.bones["{name}"].rotation_quaternion'
        location_path = f'pose.bones["{name}"].location'
        def samples(path, size, default):
            channels = [dict(zip(writer.samples.get((path, axis), [])[::2], writer.samples.get((path, axis), [])[1::2])) for axis in range(size)]
            last, values = list(default), []
            for number in numbers:
                last = [channel.get(number, last[axis]) for axis, channel in enumerate(channels)]
                values.append(last)
            return values
        rotations = [Quaternion(q).normalized() for q in samples(rotation_path, 4, (1, 0, 0, 0))]
        locations = [Vector(p) for p in samples(location_path, 3, (0, 0, 0))]
        valid = tracked[side]
        targets = list(zip(rotations, locations))
        # Clean only continuous observed segments; never average across a long
        # occlusion, invent unseen finger motion, or use Euler-angle averaging.
        for index in range(len(times)):
            if not valid[index]:
                continue
            neighbors = [index]
            for step in (-1, 1):
                j = index + step
                while 0 <= j < len(times) and valid[j] and abs(times[j] - times[index]) <= 3 * sigma:
                    if abs(times[j] - times[j - step]) > .15:
                        break
                    neighbors.append(j)
                    j += step
            weights = [(j, math.exp(-.5 * ((times[j] - times[index]) / sigma) ** 2)) for j in neighbors]
            total = sum(w for _, w in weights)
            reference = rotations[index]
            q = Quaternion(tuple(sum(rotations[j][axis] * w * (-1 if reference.dot(rotations[j]) < 0 else 1)
                                     for j, w in weights) / total for axis in range(4)))
            q.normalize()
            p = sum((locations[j] * w for j, w in weights), Vector()) / total
            targets[index] = q, p
        previous_q, previous_p = Quaternion(), Vector()
        seed_q, seed_p, reacquired = previous_q, previous_p, times[0]
        out_q, out_p = [], []
        for index, time in enumerate(times):
            dt = min(.1, time - times[index - 1]) if index else (times[1] - time if len(times) > 1 else 1 / 24)
            if valid[index]:
                goal_q, goal_p = targets[index]
                if index == 0 and valid[0]:
                    previous_q, previous_p = goal_q.copy(), goal_p.copy()
                    seed_q, seed_p, reacquired = previous_q.copy(), previous_p.copy(), time - 1
                elif index == 0 or not valid[index - 1] or time - times[index - 1] > .15:
                    seed_q, seed_p, reacquired = previous_q.copy(), previous_p.copy(), time
                blend = min(1, (time - reacquired + dt) / (.09 + .14 * strength))
                blend = blend * blend * (3 - 2 * blend)
                goal_q = seed_q.slerp(goal_q, blend) if not (index == 0 and valid[0]) else goal_q
                goal_p = seed_p.lerp(goal_p, blend) if not (index == 0 and valid[0]) else goal_p
                angle = 2 * math.acos(min(1, abs(previous_q.dot(goal_q))))
                # Allow fast finger motion, but not 180-degree one-frame snaps.
                rate = 720 if name == side + "Hand" else 1080
                alpha = min(1, math.radians(rate) * dt / max(angle, 1e-9))
                q = previous_q.slerp(goal_q, alpha)
                delta = goal_p - previous_p
                p = previous_p + delta * min(1, 1.5 * dt / max(delta.length, 1e-9))
                if previous_q.dot(q) < 0:
                    q.negate()
                previous_q, previous_p = q, p
            out_q.append(previous_q.copy())
            out_p.append(previous_p.copy())
        for path, values, size in ((rotation_path, out_q, 4), (location_path, out_p, 3)):
            for axis in range(size):
                writer.samples[(path, axis)] = [coordinate for number, value in zip(numbers, values) for coordinate in (number, value[axis])]


def solve_two_bone(start, bend, end, first_length, second_length, continuity=None, max_bend_angle=None, max_direction_angle=None):
    """Analytic IK in the tracked bend plane. Never stretch beyond reach."""
    delta = end - start
    if delta.length < 1e-5:
        delta = bend - start
    if delta.length < 1e-5:
        delta = Vector((0, 0, -1))
    direction = delta.normalized()
    if continuity is not None and "direction" in continuity and max_direction_angle is not None:
        previous = continuity["direction"]
        swing = previous.rotation_difference(direction)
        direction = Quaternion().slerp(swing, min(1, max_direction_angle / max(swing.angle, 1e-9))) @ previous
    distance = max(abs(first_length - second_length) + 1e-5,
                   min(delta.length, first_length + second_length - 1e-5))
    along = (first_length ** 2 - second_length ** 2 + distance ** 2) / (2 * distance)
    height = math.sqrt(max(0, first_length ** 2 - along ** 2))
    plane = bend - start
    plane -= direction * plane.dot(direction)
    if plane.length < 1e-5:
        plane = direction.cross(Vector((0, 1, 0)))
    if plane.length < 1e-5:
        plane = direction.cross(Vector((1, 0, 0)))
    plane.normalize()
    if continuity is not None:
        if "direction" in continuity:
            previous = continuity["direction"].rotation_difference(direction) @ continuity["plane"]
            previous -= direction * previous.dot(direction)
            if previous.length > 1e-5:
                previous.normalize()
                angle = math.atan2(direction.dot(previous.cross(plane)), previous.dot(plane))
                maximum = max_bend_angle if max_bend_angle is not None else math.pi
                plane = Quaternion(direction, max(-maximum, min(maximum, angle))) @ previous
        continuity.update(direction=direction.copy(), plane=plane.copy())
    knee = start + direction * along + plane * height
    return knee, start + direction * distance


def frame_image_context(frame, metadata, pitch, offset):
    if not frame.get("body") or (metadata.get("poseFitting") or {}).get("method") != "image-constrained-limbs-v1":
        return None
    camera = metadata["calibration"]["camera"]
    center = Vector(tuple(frame["projection"]["cameraCenter"][a] for a in ("x", "y", "z")))
    angle = math.radians(pitch)
    rotation = Matrix(((1, 0, 0), (0, -math.sin(angle), math.cos(angle)), (0, -math.cos(angle), -math.sin(angle))))
    rays = [Vector(((p["x"] * camera["width"] - camera["width"] / 2) / camera["focal"],
                    (p["y"] * camera["height"] - camera["height"] / 2) / camera["focal"], 1)) for p in frame["body"]["landmarks"]]
    return {"rays": rays, "scores": [p.get("visibility", 1) for p in frame["body"]["landmarks"]],
            "center": center,
            "to_camera": lambda p: rotation.transposed() @ (p - offset) + center,
            "to_ground": lambda p: rotation @ (p - center) + offset}


def ray_sphere(ray, parent, length):
    """Positive-depth intersections. None means the image/length constraints conflict."""
    a, b = ray.length_squared, ray.dot(parent)
    discriminant = b * b - a * (parent.length_squared - length * length)
    if discriminant < -1e-6:
        return []
    root = math.sqrt(max(0, discriminant))
    return [ray * depth for depth in ((b - root) / a, (b + root) / a) if depth > .15]


def fit_image_chain(rays, targets, lengths, previous=None, move_parent=True, center=None):
    """Exact image rays + fixed bone lengths; weak depth/temporal branch priors.

    Unlike endpoint-only 3D IK, a valid 2D elbow or wrist cannot be pulled down
    solely to accommodate a bad native depth. No solution -> explicit fallback.
    """
    def huber(value):
        return value * value if abs(value) < .2 else .4 * abs(value) - .04
    def candidates(depth):
        parent = rays[0] * depth
        for middle in ray_sphere(rays[1], parent, lengths[0]):
            for end in ray_sphere(rays[2], middle, lengths[1]):
                cost = ((depth - targets[0].z) / .15) ** 2
                cost += .03 * sum(huber(p.z - target.z) for p, target in zip((middle, end), targets[1:])) / .0225
                if previous is not None:
                    cost += 20 * sum(((p - parent) - old).length_squared for p, old in zip((middle, end), previous["relative"])) / max(.1, sum(lengths)) ** 2
                    cost += 4 * (parent.z - center.z - previous["parent"].z) ** 2 / .0225
                yield cost, (parent, middle, end)
    depths = [targets[0].z]
    if move_parent:
        depths += [targets[0].z + delta for delta in (-.2, -.1, -.05, .05, .1, .2)]
        if previous is not None:
            depth = center.z + previous["parent"].z
            if abs(depth - targets[0].z) <= .25:
                depths.append(depth)
    choices = [candidate for depth in depths if depth > .2 for candidate in candidates(depth)]
    if not choices:
        return None
    best = min(choices, key=lambda item: item[0])
    if move_parent:
        for step in (.025, .008):
            choices = [candidate for depth in (best[1][0].z - step, best[1][0].z + step)
                       if abs(depth - targets[0].z) <= .25 and depth > .2 for candidate in candidates(depth)]
            best = min([best] + choices, key=lambda item: item[0])
    return best[1]


def fit_image_shoulders(points, lengths, image, states):
    """Keep a side-on shoulder axis from collapsing/reversing its depth branch."""
    if "shoulder_width" not in lengths or not all(side + "shoulder" in points for side in ("Left", "Right")):
        return
    if min(image["scores"][i] for i in (11, 12)) < .45:
        return
    targets = [image["to_camera"](points[side + "shoulder"]) for side in ("Left", "Right")]
    rays = [image["rays"][i] for i in (11, 12)]
    previous = states.get("shoulders")
    depths = [targets[0].z + delta for delta in (-.2, -.1, -.05, 0, .05, .1, .2)]
    if previous:
        depths.append(image["center"].z + previous[0].z)
    choices = []
    for depth in depths:
        if depth <= .2:
            continue
        left = rays[0] * depth
        for right in ray_sphere(rays[1], left, lengths["shoulder_width"]):
            cost = .2 * sum((p.z - target.z) ** 2 for p, target in zip((left, right), targets))
            if previous:
                cost += 20 * sum((p - image["center"] - old).length_squared for p, old in zip((left, right), previous))
            choices.append((cost, left, right))
    if not choices:
        return
    _, left, right = min(choices, key=lambda item: item[0])
    states["shoulders"] = (left - image["center"], right - image["center"])
    points["Leftshoulder"], points["Rightshoulder"] = image["to_ground"](left), image["to_ground"](right)


def fit_image_positions(points, lengths, image, states, dt=1/24, smooth=False, contacts=None):
    fitted = set()
    fit_image_shoulders(points, lengths, image, states)
    for side, indices in (("Left", (11, 13, 15, 23, 25, 27)), ("Right", (12, 14, 16, 24, 26, 28))):
        for kind, names, labels, joints in (("arm", ("shoulder", "elbow", "wrist"), ("arm", "forearm"), indices[:3]),
                                           ("leg", ("hip", "knee", "ankle"), ("thigh", "shin"), indices[3:])):
            if kind == "leg" and (contacts or {}).get(side):
                # Support-foot targets outrank ambiguous image depth. Solving
                # the leg on a different depth branch would release its lock.
                states.pop(side + kind, None)
                continue
            keys = [side + name for name in names]
            if not all(key in points for key in keys) or not all(side + label in lengths for label in labels):
                continue
            targets = [image["to_camera"](points[key]) for key in keys]
            rays = [image["rays"][i] for i in joints]
            if kind == "leg":
                # Keep the hip/root anchor unchanged; floor constraints follow.
                rays[0] = targets[0] / max(.15, targets[0].z)
            key = side + kind
            previous = states.get(key)
            reliable = min(image["scores"][i] for i in joints) >= .45
            solution = fit_image_chain(rays, targets, [lengths[side + label] for label in labels], previous, kind == "arm", image["center"]) if reliable else None
            if solution is None and previous is not None and kind == "arm":
                # Low-confidence/contradictory rays are not new pose evidence.
                # Hold the last limb relative to its moving shoulder, not a
                # completely different depth-only IK fallback for one frame.
                solution = (targets[0], *(targets[0] + delta for delta in previous["relative"]))
            if solution is None:
                continue
            if previous is not None and kind == "arm" and smooth:
                # An image-ray solution can jump depth branches, especially on
                # re-acquisition. Bound bone directions, not bone lengths; give
                # up exact image alignment briefly instead of snapping the arm.
                parent, middle, end = solution
                old_middle, old_end = previous["relative"]
                vectors = (middle - parent, end - middle)
                old_vectors = (old_middle, old_end - old_middle)
                limited = []
                for vector, old in zip(vectors, old_vectors):
                    swing = old.normalized().rotation_difference(vector.normalized())
                    maximum = math.radians(720) * min(.1, max(.001, dt))
                    limited.append((Quaternion().slerp(swing, min(1, maximum / max(swing.angle, 1e-9))) @ old.normalized()) * vector.length)
                solution = parent, parent + limited[0], parent + limited[0] + limited[1]
            old_end = points[keys[-1]].copy()
            for name, point in zip(keys, solution):
                points[name] = image["to_ground"](point)
            shift = points[keys[-1]] - old_end
            for name in points:
                if kind == "arm" and (name.startswith(side + "finger") or name == side + "palm"):
                    points[name] += shift
                elif kind == "leg" and name in (side + "toe", side + "heel"):
                    points[name] += shift
            states[key] = {"relative": tuple(point - solution[0] for point in solution[1:]), "parent": solution[0] - image["center"]}
            fitted.add(key)
    if all(side + "shoulder" in points for side in ("Left", "Right")):
        points["chest"] = (points["Leftshoulder"] + points["Rightshoulder"]) / 2
        points["spine"] = points["hips"].lerp(points["chest"], .45)
        points["neck"] = points["chest"].lerp(points["head"], .65)
    return fitted


def fit_limb_positions(points, lengths, root_position=None, contacts=None, locks=None, bend_states=None, dt=1/24, image=None, image_states=None, model_body=False):
    """Fit fixed-length limbs to endpoints; lock reliable support feet in world space."""
    points = {key: value.copy() for key, value in points.items()}
    contacts = contacts or {}
    locks = locks if locks is not None else {}
    image_fitted = fit_image_positions(points, lengths, image, image_states if image_states is not None else {}, dt, bend_states is not None, contacts) if image else set()
    if bend_states is not None and image is None and not model_body:
        # Side-on/occluded predictions can collapse a shoulder/hip axis to zero,
        # then reverse it in one frame. Stabilize the axis before limb IK.
        for label, width in (("shoulder", "shoulder_width"), ("hip", "hip_width")):
            left, right = "Left" + label, "Right" + label
            if left not in points or right not in points or width not in lengths:
                continue
            center, delta = (points[left] + points[right]) / 2, points[left] - points[right]
            key = label + "_axis"
            previous = bend_states.get(key)
            axis = delta.normalized() if delta.length > .001 else previous or Vector((1, 0, 0))
            if previous is not None:
                if delta.length < lengths[width] * .3:
                    axis = previous.copy()
                else:
                    swing = previous.rotation_difference(axis)
                    alpha = min(1, math.radians(600) * min(.1, dt) / max(swing.angle, 1e-9))
                    axis = Quaternion().slerp(swing, alpha) @ previous
            bend_states[key] = axis.copy()
            points[left], points[right] = center + axis * lengths[width] / 2, center - axis * lengths[width] / 2
    for side in ("Left", "Right"):
        if model_body:
            # Preserve the learned model's bend and depth directions. Changing
            # lengths must not choose new elbow/knee depth branches.
            for kind, names, labels in (("arm", ("shoulder", "elbow", "wrist"), ("arm", "forearm")),
                                        ("leg", ("hip", "knee", "ankle"), ("thigh", "shin"))):
                keys = [side + name for name in names]
                if not all(key in points for key in keys) or not all(side + label in lengths for label in labels):
                    continue
                start, middle, end = [points[key].copy() for key in keys]
                upper, lower = middle - start, end - middle
                if min(upper.length, lower.length) < .001:
                    continue
                points[keys[1]] = start + upper.normalized() * lengths[side + labels[0]]
                points[keys[2]] = points[keys[1]] + lower.normalized() * lengths[side + labels[1]]
                shift = points[keys[2]] - end
                for name in points:
                    if (kind == "arm" and (name.startswith(side + "finger") or name == side + "palm")) or (kind == "leg" and name in (side + "toe", side + "heel")):
                        points[name] += shift
                image_fitted.add(side + kind)
        if side + "arm" not in image_fitted and all(side + key in points for key in ("shoulder", "elbow", "wrist")) and all(side + key in lengths for key in ("arm", "forearm")):
            old_wrist = points[side + "wrist"].copy()
            elbow, wrist = solve_two_bone(points[side + "shoulder"], points[side + "elbow"], old_wrist,
                                          lengths[side + "arm"], lengths[side + "forearm"],
                                          bend_states.setdefault(side + "arm", {}) if bend_states is not None else None,
                                          math.radians(600) * min(.1, dt),
                                          math.radians(600) * min(.1, dt) if bend_states is not None else None)
            points[side + "elbow"], points[side + "wrist"] = elbow, wrist
            shift = wrist - old_wrist
            for name in points:
                if name.startswith(side + "finger") or name == side + "palm":
                    points[name] += shift
        if not all(side + key in points for key in ("hip", "knee", "ankle", "toe")) or not all(side + key in lengths for key in ("thigh", "shin", "foot")):
            locks.pop(side, None)
            continue
        ankle, toe = points[side + "ankle"].copy(), points[side + "toe"].copy()
        foot_vector = toe - ankle
        if foot_vector.length < 1e-5:
            foot_vector = Vector((0, -.1, 0))
        foot_vector = foot_vector.normalized() * lengths[side + "foot"]
        if model_body and bend_states is not None:
            # Foot tips come from framewise RTMW3D, not the learned body17.
            # Bound their independent depth flips without filtering the model's
            # ankle/knee/hip directions or releasing a support-foot lock.
            old = bend_states.get(side + "native_foot")
            if old is not None:
                swing = old.normalized().rotation_difference(foot_vector.normalized())
                alpha = min(1, math.radians(720) * min(.1, max(.001, dt)) / max(swing.angle, 1e-9))
                foot_vector = (Quaternion().slerp(swing, alpha) @ old.normalized()) * lengths[side + "foot"]
        if root_position is not None:
            if not contacts.get(side):
                locks.pop(side, None)
            world_ankle = ankle + root_position
            if contacts.get(side):
                lock = locks.get(side)
                if lock and (world_ankle - lock[0]).length > .12:
                    locks.pop(side, None)
                    lock = None
                if lock is None:
                    locks[side] = (world_ankle.copy(), foot_vector.copy())
                ankle = locks[side][0] - root_position
                foot_vector = locks[side][1].copy()
            # The rendered foot's tip must not sink, even if the source heel is
            # lower than the toe or the median foot is longer than the first pose.
            penetration = -(ankle + foot_vector + root_position).z
            if penetration > 0:
                ankle.z += penetration
        if model_body and bend_states is not None:
            bend_states[side + "native_foot"] = foot_vector.copy()
        if model_body and side + "leg" in image_fitted and (ankle - points[side + "ankle"]).length < 1e-7:
            knee = points[side + "knee"].copy()
        else:
            knee, ankle = solve_two_bone(points[side + "hip"], points[side + "knee"], ankle,
                                        lengths[side + "thigh"], lengths[side + "shin"],
                                        bend_states.setdefault(side + "leg", {}) if bend_states is not None and side + "leg" not in image_fitted else None,
                                        math.radians(600) * min(.1, dt))
        points[side + "knee"], points[side + "ankle"] = knee, ankle
        points[side + "toe"] = ankle + foot_vector
    return points


def reference_orientation(direction, lateral):
    y = direction.normalized()
    x = lateral - y * lateral.dot(y)
    if x.length < 1e-5:
        return None
    x.normalize()
    z = x.cross(y).normalized()
    return Matrix((x, y, z)).transposed().to_quaternion()


def stabilize_reference(desired, direction, previous, maximum):
    """Limit ambiguous measured axial roll while still aiming at the fitted joint."""
    transported = transport_orientation(direction, desired, previous)
    axis = direction.normalized()
    before, after = transported @ Vector((1, 0, 0)), desired @ Vector((1, 0, 0))
    angle = math.atan2(axis.dot(before.cross(after)), before.dot(after))
    return Quaternion(axis, max(-maximum, min(maximum, angle))) @ transported


def lateral_axis(name, points):
    # Aim alone cannot recover pelvis/chest yaw. A second measured axis can.
    if name in ("Hips", "Spine") and all(key in points for key in ("Lefthip", "Righthip")):
        return points["Lefthip"] - points["Righthip"]
    if name in ("Chest", "Neck", "Head") and all(key in points for key in ("Leftshoulder", "Rightshoulder")):
        return points["Leftshoulder"] - points["Rightshoulder"]
    for side in ("Left", "Right"):
        if name == side + "Hand" and all(side + "finger" + str(i) in points for i in (5, 17)):
            return points[side + "finger5"] - points[side + "finger17"]
    return None


def positions_for_frame(frame, offset, anchors, pitch=0, centered=False, profiles=None):
    result = body_positions(frame.get("body"), offset, pitch, centered)
    if profiles is not None:
        for side in profiles:
            # Hold the local hand/finger pose during rejected/missing detections,
            # while its parent forearm continues to follow the body.
            result.pop(side + "palm", None)
    for hand in frame.get("hands", []):
        side = hand["side"]
        if profiles is not None and (side not in profiles or not usable_hand(hand, profiles[side])):
            continue
        points = [converted(p, pitch) for p in hand["world"]]
        if profiles is not None:
            points = fit_hand_positions(points, profiles[side]["lengths"])
        anchor = result.get(side + "wrist", anchors[side])
        for index, point in enumerate(points):
            result[f"{side}finger{index}"] = point - points[0] + anchor
        result[side + "wrist"] = anchor
        result[side + "palm"] = result[f"{side}finger9"]
    return result


class CurveWriter:
    """Bulk keyframe insertion, including Blender's newer layered Actions API."""
    def __init__(self, target, name):
        self.target = target
        self.name = name
        self.samples = {}

    def add(self, path, index, frame, value):
        self.samples.setdefault((path, index), []).extend((float(frame), float(value)))

    def flush(self):
        if not self.samples:
            return
        action = bpy.data.actions.new(self.name)
        self.target.animation_data_create()
        self.target.animation_data.action = action
        if bpy.app.version >= (4, 4, 0):
            slot = action.slots.new(id_type=self.target.id_type, name=self.target.name)
            self.target.animation_data.action_slot = slot
            layer = action.layers.new("Capture")
            strip = layer.strips.new(type="KEYFRAME")
            curves = strip.channelbag(slot, ensure=True).fcurves
        else:
            curves = action.fcurves
        for (path, index), coordinates in self.samples.items():
            curve = curves.new(data_path=path, index=index)
            curve.keyframe_points.add(len(coordinates) // 2)
            curve.keyframe_points.foreach_set("co", coordinates)
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
            curve.update_autoflags(self.target)
            curve.update()
        self.samples.clear()


def create_source_view(collection, frames, metadata, pitch, offset, root_object, start_frame):
    """Pose-comparison camera, NOT a recovered world/camera trajectory.

    Root stabilization changes global translation, so the review camera tracks
    that adjustment. It leaves the actual armature and its root keys untouched.
    Manual imports do not replace the scene's active camera or render settings.
    """
    if not metadata.get("poseFitting") and not metadata.get("bodyEstimator"):
        return None
    intrinsics = metadata["calibration"]["camera"]
    data = bpy.data.cameras.new("Framecraft Source View")
    camera = bpy.data.objects.new("Framecraft Source View", data)
    collection.objects.link(camera)
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = 36 * intrinsics["focal"] / intrinsics["width"]
    data.clip_start = .01
    data.clip_end = 200
    orthographic = bool(metadata.get("bodyEstimator"))
    if orthographic:
        data.type = "ORTHO"
    angle = math.radians(pitch)
    rotation = Matrix(((1, 0, 0), (0, -math.sin(angle), math.cos(angle)), (0, -math.cos(angle), -math.sin(angle))))
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (rotation @ Matrix.Diagonal((1, -1, -1))).to_quaternion()
    camera["capture_review_only"] = True
    camera["capture_notes"] = "Animated pose-comparison view matching estimated source intrinsics. Compensates approximate root stabilization; NOT recovered camera motion. Ground travel remains on Framecraft Root."
    writer = CurveWriter(camera, "Framecraft Source Comparison")
    scale_writer = CurveWriter(data, "Framecraft Source Scale")
    center, root = None, Vector()
    for frame in frames:
        if frame.get("projection"):
            center = Vector(tuple(frame["projection"]["cameraCenter"][a] for a in ("x", "y", "z")))
            if orthographic:
                center.z += 5  # Orthographic depth cannot change image scale; keep all joints in front.
                scale_writer.add("ortho_scale", 0, start_frame + frame["time"] * metadata["fps"], intrinsics["width"] / frame["projection"]["pixelsPerMeter"])
        if root_object:
            root = Vector(root_location(frame, root))
        if center is not None:
            location = root + offset - rotation @ center
            for axis in range(3):
                writer.add("location", axis, start_frame + frame["time"] * metadata["fps"], location[axis])
    writer.flush()
    scale_writer.flush()
    return camera


def face_positions(face, frame, metadata, offset, last_anchor, pitch=0, centered=False):
    if face.get("topology") == "coco-wholebody-68":
        # RTMW3D face is already in the same metric, hip-relative camera space
        # as the body. Do not apply MediaPipe's eye-width heuristic a second time.
        coords = [converted(point, pitch) + offset for point in face["world"]]
        return coords, coords[30]
    points = face["landmarks"]
    nose = points[1]
    # Approximate scale from eye spacing; monocular face depth is not metric.
    aspect = metadata["height"] / metadata["width"]
    eye_width = math.hypot(points[33]["x"] - points[263]["x"],
                           (points[33]["y"] - points[263]["y"]) * aspect)
    scale = min(5.0, 0.09 / max(eye_width, 0.015))
    body = body_positions(frame.get("body"), offset, pitch, centered)
    anchor = body.get("nose", last_anchor)
    coords = [converted({"x": (p["x"] - nose["x"]) * scale,
                         "y": (p["y"] - nose["y"]) * scale * aspect,
                         "z": (p["z"] - nose["z"]) * scale}, pitch) + anchor for p in points]
    return coords, anchor


def gvhmr_joint_matrices(frame):
    """Body-joint FK in root-relative Z-up space, preserving measured twist."""
    result = []
    for index, (axis_angle, offset) in enumerate(zip(frame["rotations"], frame["offsets"])):
        axis = Vector(axis_angle)
        rotation = Quaternion(axis.normalized(), axis.length) if axis.length > 1e-8 else Quaternion()
        local = rotation.to_matrix().to_4x4()
        local.translation = Vector(offset)
        parent = GVHMR_PARENTS[index]
        result.append(result[parent] @ local if parent >= 0 else local)
    return result


def import_gvhmr_capture(context, data, start_frame=1, frame_step=1, include_root=True):
    validate_gvhmr_capture(data)
    if not isinstance(frame_step, int) or frame_step < 1:
        raise ValueError("Frame step must be a positive integer.")
    frames, scene = data["frames"], context.scene
    collection = bpy.data.collections.new("Framecraft GVHMR Capture")
    scene.collection.children.link(collection)
    root = bpy.data.objects.new("Framecraft GVHMR Root", None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = .25
    collection.objects.link(root)
    root["capture_notes"] = "GVHMR world pelvis translation in estimated meters. Body rotations are on the child rig. No additional MediaPipe root reconstruction."
    root.location = frames[0]["root"]
    if include_root:
        writer = CurveWriter(root, "Framecraft GVHMR World Travel")
        for frame in frames:
            for axis in range(3):
                writer.add("location", axis, start_frame + frame["time"] * data["metadata"]["fps"], frame["root"][axis])
        writer.flush()
    armature = bpy.data.armatures.new("Framecraft GVHMR Skeleton")
    rig = bpy.data.objects.new("Framecraft GVHMR Rig", armature)
    collection.objects.link(rig)
    rig.parent = root
    rig.show_in_front = True
    rig["capture_engine"] = "GVHMR"
    rig["capture_source"] = data["metadata"].get("filename", "video")
    rig["capture_notes"] = "Native 22-joint body rotations, including twist, and per-frame model offsets. No fingers/face inferred. Retarget separately."
    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    context.view_layer.objects.active = rig
    rest = [Vector(p) for p in data["skeleton"]["restJoints"]]
    # Choose a representative child for display only. Animation uses all 22
    # joint frames; bone display directions must not discard axial rotations.
    tail_children = {0: 3, 3: 6, 6: 9, 9: 12}
    bpy.ops.object.mode_set(mode="EDIT")
    for index, name in enumerate(GVHMR_NAMES):
        bone = armature.edit_bones.new(name)
        bone.head = rest[index]
        children = [j for j, p in enumerate(GVHMR_PARENTS) if p == index]
        parent = GVHMR_PARENTS[index]
        if children:
            bone.tail = rest[tail_children.get(index, children[0])]
        elif parent >= 0 and (rest[index] - rest[parent]).length > 1e-5:
            bone.tail = rest[index] + (rest[index] - rest[parent]).normalized() * .1
        else:
            bone.tail = bone.head + Vector((0, 0, .1))
        if (bone.tail - bone.head).length < 1e-5:
            bone.tail = bone.head + Vector((0, 0, .01))
        if parent >= 0:
            bone.parent = armature.edit_bones[GVHMR_NAMES[parent]]
    bpy.ops.object.mode_set(mode="OBJECT")
    rest_rotations = [armature.bones[name].matrix_local.to_quaternion().to_matrix().to_4x4() for name in GVHMR_NAMES]
    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"
    writer = CurveWriter(rig, "Framecraft GVHMR Body Rotations")
    selected = list(range(0, len(frames), frame_step))
    if selected[-1] != len(frames) - 1:
        selected.append(len(frames) - 1)
    previous = {}
    context.window_manager.progress_begin(0, len(selected))
    try:
        for progress, index in enumerate(selected):
            frame = frames[index]
            matrices = [joint @ rest_rotation for joint, rest_rotation in zip(gvhmr_joint_matrices(frame), rest_rotations)]
            number = start_frame + frame["time"] * data["metadata"]["fps"]
            for joint, name in enumerate(GVHMR_NAMES):
                bone = armature.bones[name]
                parent = GVHMR_PARENTS[joint]
                kwargs = {"parent_matrix": matrices[parent], "parent_matrix_local": bone.parent.matrix_local} if parent >= 0 else {}
                basis = bone.convert_local_to_pose(matrices[joint], bone.matrix_local, invert=True, **kwargs)
                location, rotation, _scale = basis.decompose()
                if name in previous and previous[name].dot(rotation) < 0:
                    rotation.negate()
                previous[name] = rotation.copy()
                for axis in range(3):
                    writer.add(f'pose.bones["{name}"].location', axis, number, location[axis])
                for axis in range(4):
                    writer.add(f'pose.bones["{name}"].rotation_quaternion', axis, number, rotation[axis])
            context.window_manager.progress_update(progress)
        writer.flush()
    finally:
        context.window_manager.progress_end()
    fps = data["metadata"]["fps"]
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    scene.frame_start = min(scene.frame_start, start_frame)
    scene.frame_end = max(scene.frame_end, math.ceil(start_frame + frames[-1]["time"] * fps))
    scene.frame_set(start_frame)
    return collection


def import_capture(context, data, start_frame=1, frame_step=1, include_face=True, include_root=True, improve_fit=True, smooth_motion=True):
    validate_capture(data)
    if data.get("format") == "framecraft-gvhmr":
        return import_gvhmr_capture(context, data, start_frame, frame_step, include_root)
    frames = data["frames"]
    metadata = data["metadata"]
    model_body = (metadata.get("bodyEstimator") or {}).get("model") == "motionagformer-b"
    temporal_strength = 0
    if smooth_motion and improve_fit and metadata.get("engine") == "rtmw3d-x":
        requested = (metadata.get("animationSmoothing") or {}).get("strength", (metadata.get("refinement") or {}).get("strength", .35))
        if not isinstance(requested, (int, float)) or not math.isfinite(requested) or not 0 <= requested <= .85:
            raise ValueError("Animation smoothing strength must be 0–0.85.")
        temporal_strength = requested
    lengths = segment_lengths(frames) if improve_fit else {}
    profiles = hand_profiles(frames, metadata)
    if not model_body:
        frames = smooth_body_frames(frames, temporal_strength, metadata.get("heightMeters", 1.7) / 1.7)
    scene = context.scene
    rooted = include_root and bool((data.get("rootMotion") or {}).get("enabled"))
    pitch = data["rootMotion"]["pitch"] if rooted else 0
    # Do not rename, clear, or replace the user's existing scene objects.
    collection = bpy.data.collections.new("Framecraft Capture")
    scene.collection.children.link(collection)
    first_body_frame = next((f for f in frames if f.get("body")), None)
    first_body = first_body_frame["body"] if first_body_frame else None
    offset = Vector((0, 0, 0))
    if first_body and not rooted:
        foot_z = min(converted(first_body["world"][i]).z for i in (27, 28, 29, 30, 31, 32))
        offset.z = -foot_z + 0.03
    rest_points = body_positions(first_body, offset, pitch, rooted)
    if improve_fit:
        rest_image = frame_image_context(first_body_frame, metadata, pitch, offset) if first_body_frame else None
        rest_points = fit_limb_positions(rest_points, lengths, bend_states={} if temporal_strength else None, image=rest_image, model_body=model_body)
    root_object = None
    if rooted:
        root_object = bpy.data.objects.new("Framecraft Root", None)
        collection.objects.link(root_object)
        root_object.empty_display_type = "ARROWS"
        root_object.empty_display_size = 0.25
        root_object["capture_notes"] = "Estimated ground-based root translation, in meters. Z is hip height above floor. Fixed-camera assumption; no scene/camera tracking."
        root_writer = CurveWriter(root_object, "Framecraft Ground Root Motion")
        fallback = next((root_location(f, (0, 0, 1)) for f in frames if f.get("root")), (0, 0, 1))
        for frame in frames:
            fallback = root_location(frame, fallback)
            number = start_frame + frame["time"] * metadata["fps"]
            for axis in range(3):
                root_writer.add("location", axis, number, fallback[axis])
        root_writer.flush()
        # Non-rendering floor guide; does not change an existing scene's floor.
        points = [root_location(f, (0, 0, 0)) for f in frames if f.get("root")]
        min_x, max_x = min(p[0] for p in points) - 1, max(p[0] for p in points) + 1
        min_y, max_y = min(p[1] for p in points) - 1, max(p[1] for p in points) + 1
        mesh = bpy.data.meshes.new("Framecraft Ground Guide")
        mesh.from_pydata([(min_x, min_y, 0), (max_x, min_y, 0), (max_x, max_y, 0), (min_x, max_y, 0)],
                         [(0, 1), (1, 2), (2, 3), (3, 0)], [])
        ground = bpy.data.objects.new("Framecraft Ground Reference", mesh)
        ground.hide_render = True
        collection.objects.link(ground)
    anchors = {side: rest_points.get(side + "wrist", Vector((x, 0, 1.2)))
               for side, x in (("Left", -0.25), ("Right", 0.25))}
    specs = list(BODY_BONES) if first_body else []
    for side in ("Left", "Right"):
        if side not in profiles:
            continue
        hand_points = positions_for_frame(profiles[side]["frame"], offset, anchors, pitch, rooted, profiles)
        # Recenter a reliable hand pose at the rest wrist, using clip-wide lengths.
        shift = anchors[side] - hand_points[side + "wrist"]
        for key, value in hand_points.items():
            if key.startswith(side + "finger"):
                rest_points[key] = value + shift
        rest_points[side + "wrist"] = anchors[side]
        rest_points[side + "palm"] = rest_points[f"{side}finger9"]
        if not first_body:
            specs.append((side + "Hand", side + "wrist", side + "palm", None))
        for finger, index in FINGER_CHAINS:
            parent = side + "Hand"
            for segment in range(3):
                name = f"{side}{finger}{segment + 1}"
                specs.append((name, f"{side}finger{index + segment}", f"{side}finger{index + segment + 1}", parent))
                parent = name

    rig = None
    writer = None
    if specs:
        armature = bpy.data.armatures.new("Framecraft Skeleton")
        rig = bpy.data.objects.new("Framecraft Rig", armature)
        collection.objects.link(rig)
        if root_object:
            rig.parent = root_object
        rig.show_in_front = True
        rig["capture_source"] = metadata.get("filename", "video")
        rig["capture_engine"] = metadata.get("engine", "mediapipe")
        rig["capture_body_model"] = "MotionAGFormer-B temporal body" if model_body else "RTMW3D framewise body" if metadata.get("engine") == "rtmw3d-x" else "MediaPipe"
        rig["capture_importer_version"] = ".".join(map(str, bl_info["version"]))
        rig["capture_hand_fitting"] = "Geometry/confidence gate; clip-median finger/palm lengths; connected finger chains. Rejected hands hold their last local pose."
        rig["capture_temporal_smoothing"] = temporal_strength
        rig["capture_image_fitting"] = (metadata.get("poseFitting") or {}).get("method", "legacy-3d-targets")
        rig["capture_temporal_notes"] = "Centered pose-target cleanup, minimum-twist aiming, quaternion hand filtering and eased re-acquisition; fixed-length IK follows cleanup. Root/face samples are unchanged." if temporal_strength else "Temporal rig cleanup disabled."
        if model_body:
            rig["capture_temporal_notes"] = "MotionAGFormer-B overlapping temporal inference; learned 3D body directions retained. No image-ray branch solver or additional body-target smoothing. RTMW3D hand cleanup and bounded native foot-tip directions remain separate."
            if metadata["bodyEstimator"].get("refinement"):
                rig["capture_temporal_notes"] = "Observed arm vectors corrected using detector confidence/agreement and neural depth priors; continuous sequence depth transitions. Neural torso/legs retained; native hand/foot cleanup is separate. No additional body-target smoothing."
        rig["capture_root_status"] = "Ground travel is animated on parent Framecraft Root" if rooted else "IN PLACE: root track absent or root import disabled"
        rig["capture_fitting"] = "Median limb lengths; two-bone fitting; measured torso yaw; support-foot anchors" if improve_fit else "Legacy aim-only fitting"
        if model_body and improve_fit:
            rig["capture_fitting"] = "Body17 directions (observed-arm refinement when present) with clip-median limb lengths; support-foot constraints. No importer image-ray depth-branch fitting."
        rig["capture_notes"] = ("Ground travel is on parent Framecraft Root. " if rooted else "In-place single-camera estimates. ") + "Missing tracks hold their last pose. Bone twist and camera trajectory are not recovered. Retarget separately."
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        rig.select_set(True)
        context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for name, head, tail, parent in specs:
            bone = armature.edit_bones.new(name)
            bone.head = rest_points[head]
            bone.tail = rest_points[tail]
            if (bone.tail - bone.head).length < 0.001:
                bone.tail = bone.head + Vector((0, 0, 0.01))
            if parent:
                bone.parent = armature.edit_bones[parent]
        bpy.ops.object.mode_set(mode="OBJECT")
        for bone in rig.pose.bones:
            bone.rotation_mode = "QUATERNION"
        writer = CurveWriter(rig, "Framecraft Body and Fingers")

    face_object = None
    face_writer = None
    scores_writer = None
    first_face_frame = next((f for f in frames if f.get("face")), None)
    face_anchor = rest_points.get("nose", Vector((0, 0, 1.65)))
    if include_face and first_face_frame:
        coords, face_anchor = face_positions(first_face_frame["face"], first_face_frame, metadata, offset, face_anchor, pitch, rooted)
        mesh = bpy.data.meshes.new("Framecraft Facial Landmarks")
        mesh.from_pydata(coords, data.get("faceEdges", []), [])
        mesh.update()
        face_object = bpy.data.objects.new("Framecraft Face", mesh)
        collection.objects.link(face_object)
        if root_object:
            face_object.parent = root_object
        face_object.show_in_front = True
        face_object["capture_notes"] = ("Native RTMW3D face68 wire mesh; no expression scores or character face rig."
                                         if data.get("faceTopology") == "coco-wholebody-68" else
                                         "Animated facial wire mesh, not a character face rig. fc_* properties are expression scores for manual mapping to character shape keys.")
        face_writer = CurveWriter(mesh, "Framecraft Face Landmarks")
        scores_writer = CurveWriter(face_object, "Framecraft Expression Scores")

    previous_world = {}
    previous_basis = {}
    previous_location = {}
    previous_matrices = {}
    foot_locks = {}
    bend_states = {} if temporal_strength else None
    image_states = {}
    previous_root_time = None
    if rig:
        rest_rot = {b.name: b.matrix_local.to_quaternion() for b in rig.data.bones}
        rest_dir = {b.name: (b.tail_local - b.head_local).normalized() for b in rig.data.bones}
        rest_reference = {name: reference_orientation(rest_dir[name], axis) for name, *_ in specs
                          if (axis := lateral_axis(name, rest_points)) is not None}
    selected = list(range(0, len(frames), frame_step))
    if selected[-1] != len(frames) - 1:
        selected.append(len(frames) - 1)
    animation_times, animation_numbers = [], []
    hand_tracked = {side: [] for side in profiles}
    context.window_manager.progress_begin(0, len(selected))
    try:
        for progress, index in enumerate(selected):
            frame = frames[index]
            number = start_frame + frame["time"] * metadata["fps"]
            points = positions_for_frame(frame, offset, anchors, pitch, rooted, profiles)
            animation_times.append(frame["time"])
            animation_numbers.append(number)
            for side in hand_tracked:
                hand_tracked[side].append(side + "finger0" in points)
            root_sample = frame.get("root") if rooted else None
            dt = frame["time"] - previous_root_time if previous_root_time is not None else 1 / metadata["fps"]
            if previous_root_time is None or frame["time"] - previous_root_time > .15 or not frame.get("body"):
                foot_locks.clear()
                if bend_states is not None:
                    bend_states.clear()
                image_states.clear()
            previous_root_time = frame["time"]
            if improve_fit:
                root_position = Vector(root_location(frame, (0, 0, 0))) if root_sample else None
                contacts = root_sample.get("contacts", {}) if root_sample and root_sample.get("status") == "tracked" else {}
                image = frame_image_context(frame, metadata, pitch, offset)
                points = fit_limb_positions(points, lengths, root_position, contacts, foot_locks, bend_states, dt, image, image_states, model_body)
            if rig:
                for name, head, tail, parent in specs:
                    if head not in points or tail not in points:
                        # Explicit keys keep missing detections at the last known pose.
                        if name in previous_basis:
                            for axis in range(4):
                                writer.add(f'pose.bones["{name}"].rotation_quaternion', axis, number, previous_basis[name][axis])
                            if name in previous_location:
                                for axis in range(3):
                                    writer.add(f'pose.bones["{name}"].location', axis, number, previous_location[name][axis])
                        continue
                    direction = points[tail] - points[head]
                    if direction.length < 0.001:
                        continue
                    desired = rest_dir[name].rotation_difference(direction.normalized()) @ rest_rot[name]
                    if temporal_strength:
                        desired = transport_orientation(direction, rest_rot[name], previous_world.get(name))
                    if improve_fit and name in rest_reference and rest_reference[name] is not None:
                        axis = lateral_axis(name, points)
                        reference = reference_orientation(direction, axis) if axis is not None else None
                        if reference is not None:
                            desired = reference @ rest_reference[name].inverted() @ rest_rot[name]
                            if temporal_strength and name in previous_world:
                                desired = stabilize_reference(desired, direction, previous_world[name], math.radians(600) * min(.1, dt))
                    if parent:
                        relative_rest = rest_rot[parent].inverted() @ rest_rot[name]
                        parent_world = previous_world.get(parent, rest_rot[parent])
                        basis = relative_rest.inverted() @ parent_world.inverted() @ desired
                    else:
                        basis = rest_rot[name].inverted() @ desired
                        location = rest_rot[name].inverted() @ (points[head] - rest_points[head])
                        if not improve_fit:
                            previous_location[name] = location.copy()
                            for axis in range(3):
                                writer.add(f'pose.bones["{name}"].location', axis, number, location[axis])
                    if improve_fit:
                        # Match actual joint heads, not just directions. The old
                        # rest offsets drifted when the pelvis or shoulders turned.
                        desired_matrix = desired.to_matrix().to_4x4()
                        desired_matrix.translation = points[head]
                        bone = rig.data.bones[name]
                        kwargs = {"parent_matrix": previous_matrices.get(parent, rig.data.bones[parent].matrix_local),
                                  "parent_matrix_local": rig.data.bones[parent].matrix_local} if parent else {}
                        local_matrix = bone.convert_local_to_pose(desired_matrix, bone.matrix_local, invert=True, **kwargs)
                        basis = local_matrix.to_quaternion()
                        location = local_matrix.translation
                        previous_location[name] = location.copy()
                        previous_matrices[name] = desired_matrix
                        for axis in range(3):
                            path = f'pose.bones["{name}"].location'
                            writer.add(path, axis, number, location[axis])
                    basis.normalize()
                    if name in previous_basis and basis.dot(previous_basis[name]) < 0:
                        basis.negate()
                    previous_basis[name] = basis.copy()
                    previous_world[name] = desired
                    for axis in range(4):
                        writer.add(f'pose.bones["{name}"].rotation_quaternion', axis, number, basis[axis])
            if face_object:
                face = frame.get("face")
                # Face gaps are visible as gaps; body/finger rotations hold their last keys.
                for path in ("hide_viewport", "hide_render"):
                    scores_writer.add(path, 0, number, 0 if face else 1)
                if face:
                    coords, face_anchor = face_positions(face, frame, metadata, offset, face_anchor, pitch, rooted)
                    for vertex_index, point in enumerate(coords):
                        for axis in range(3):
                            face_writer.add(f"vertices[{vertex_index}].co", axis, number, point[axis])
                    for name, value in face.get("blendshapes", {}).items():
                        prop = "fc_" + name
                        face_object[prop] = value
                        face_object.id_properties_ui(prop).update(min=0.0, max=1.0)
                        scores_writer.add(f'["{prop}"]', 0, number, value)
            context.window_manager.progress_update(progress)
        if writer:
            smooth_hand_animation(writer, specs, animation_times, animation_numbers, hand_tracked, temporal_strength)
            writer.flush()
        if face_writer:
            face_writer.flush()
            scores_writer.flush()
    finally:
        context.window_manager.progress_end()
    fps = metadata["fps"]
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    end = math.ceil(start_frame + frames[-1]["time"] * fps)
    scene.frame_start = min(scene.frame_start, start_frame)
    scene.frame_end = max(scene.frame_end, end)
    create_source_view(collection, frames, metadata, pitch, offset, root_object, start_frame)
    scene.frame_set(start_frame)
    return collection


if bpy is not None:
    class FRAMECRAFT_OT_import(bpy.types.Operator, ImportHelper):
        bl_idname = "import_scene.framecraft_capture"
        bl_label = "Import Framecraft Capture"
        bl_options = {"REGISTER", "UNDO"}
        filename_ext = ".json"
        filter_glob: StringProperty(default="*.mocap.json;*.gvhmr.json;*.json", options={"HIDDEN"})
        start_frame: IntProperty(name="Start frame", default=1, min=1)
        frame_step: IntProperty(name="Import every Nth sample", default=1, min=1, max=12)
        include_face: BoolProperty(name="Import facial wire mesh and expression scores", default=True)
        include_root: BoolProperty(name="Import estimated ground root motion", default=True)
        improve_fit: BoolProperty(name="Fit landmark limbs, torso rotation, and support feet (not GVHMR)", default=True)
        smooth_motion: BoolProperty(name="Smooth RTMW3D pose timing and hand re-acquisition", default=True)

        def execute(self, context):
            try:
                path = Path(self.filepath)
                if path.stat().st_size > 512 * 1024 * 1024:
                    raise ValueError("Capture exceeds the 512 MB import limit.")
                data = json.loads(path.read_text(encoding="utf-8"))
                validate_capture(data)
                import_capture(context, data, self.start_frame, self.frame_step, self.include_face, self.include_root, self.improve_fit, self.smooth_motion)
            except Exception as error:
                self.report({"ERROR"}, f"Import failed: {error}. Undo if any capture objects were created.")
                return {"CANCELLED"}
            summary = root_motion_summary(data) if self.include_root else None
            if data.get("format") == "framecraft-gvhmr":
                self.report({"INFO"}, "GVHMR native body rotations imported. " +
                            ("World travel is on Framecraft GVHMR Root." if self.include_root else "World travel disabled; first pelvis position retained."))
                return {"FINISHED"}
            if self.include_root and not summary:
                self.report({"WARNING"}, "Imported IN PLACE: this file has no root track. In Framecraft, use 'Repair older capture JSON' or re-extract with root motion enabled.")
            elif summary and math.hypot(summary["x"], summary["y"]) < 0.02:
                self.report({"WARNING"}, "Root data is present but estimates less than 2 cm of ground travel. Check the clip, camera settings, and foot stabilization.")
            elif summary:
                self.report({"INFO"}, f"Root travel imported: X range {summary['x']:.2f} m, Y range {summary['y']:.2f} m. Play the timeline; Framecraft Rig follows Framecraft Root.")
            else:
                self.report({"INFO"}, "Capture imported in place (root import disabled).")
            return {"FINISHED"}

    def menu_import(self, _context):
        self.layout.operator(FRAMECRAFT_OT_import.bl_idname, text="Framecraft Capture (.mocap.json / .gvhmr.json)")

    def register():
        namespace = bpy.app.driver_namespace
        old = namespace.get("framecraft_operator") or bpy.types.Operator.bl_rna_get_subclass_py("IMPORT_SCENE_OT_framecraft_capture", None)
        if old:
            bpy.utils.unregister_class(old)
        # When rerunning in the Text Editor, remove the previous menu callback.
        previous_menu = namespace.get("framecraft_menu")
        if previous_menu:
            bpy.types.TOPBAR_MT_file_import.remove(previous_menu)
        bpy.utils.register_class(FRAMECRAFT_OT_import)
        bpy.types.TOPBAR_MT_file_import.append(menu_import)
        namespace["framecraft_menu"] = menu_import
        namespace["framecraft_operator"] = FRAMECRAFT_OT_import

    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_import)
        bpy.utils.unregister_class(FRAMECRAFT_OT_import)
        bpy.app.driver_namespace.pop("framecraft_menu", None)
        bpy.app.driver_namespace.pop("framecraft_operator", None)

    if __name__ == "__main__":
        register()
