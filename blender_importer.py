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
    "version": (0, 5, 0), "blender": (4, 2, 0),
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
    face68 = data.get("faceTopology") == "coco-wholebody-68"
    if data.get("faceTopology") not in (None, "coco-wholebody-68") or (face68 and metadata.get("engine") != "rtmw3d-x"):
        raise ValueError("Unsupported facial topology/engine.")
    face_count = 68 if face68 else 478
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
    for side, indices in (("Left", (11, 13, 15, 19, 23, 25, 27, 31)),
                          ("Right", (12, 14, 16, 20, 24, 26, 28, 32))):
        for name, index in zip(("shoulder", "elbow", "wrist", "palm", "hip", "knee", "ankle", "toe"), indices):
            result[side + name] = p[index]
    result["Leftheel"], result["Rightheel"] = p[29], p[30]
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
    return result


def solve_two_bone(start, bend, end, first_length, second_length):
    """Analytic IK in the tracked bend plane. Never stretch beyond reach."""
    delta = end - start
    if delta.length < 1e-5:
        delta = bend - start
    if delta.length < 1e-5:
        delta = Vector((0, 0, -1))
    direction = delta.normalized()
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
    knee = start + direction * along + plane.normalized() * height
    return knee, start + direction * distance


def fit_limb_positions(points, lengths, root_position=None, contacts=None, locks=None):
    """Fit fixed-length limbs to endpoints; lock reliable support feet in world space."""
    points = {key: value.copy() for key, value in points.items()}
    contacts = contacts or {}
    locks = locks if locks is not None else {}
    for side in ("Left", "Right"):
        if all(side + key in points for key in ("shoulder", "elbow", "wrist")) and all(side + key in lengths for key in ("arm", "forearm")):
            old_wrist = points[side + "wrist"].copy()
            elbow, wrist = solve_two_bone(points[side + "shoulder"], points[side + "elbow"], old_wrist,
                                          lengths[side + "arm"], lengths[side + "forearm"])
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
        knee, ankle = solve_two_bone(points[side + "hip"], points[side + "knee"], ankle,
                                    lengths[side + "thigh"], lengths[side + "shin"])
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


def positions_for_frame(frame, offset, anchors, pitch=0, centered=False):
    result = body_positions(frame.get("body"), offset, pitch, centered)
    for hand in frame.get("hands", []):
        side = hand["side"]
        points = [converted(p, pitch) for p in hand["world"]]
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


def import_capture(context, data, start_frame=1, frame_step=1, include_face=True, include_root=True, improve_fit=True):
    validate_capture(data)
    if data.get("format") == "framecraft-gvhmr":
        return import_gvhmr_capture(context, data, start_frame, frame_step, include_root)
    frames = data["frames"]
    metadata = data["metadata"]
    scene = context.scene
    rooted = include_root and bool((data.get("rootMotion") or {}).get("enabled"))
    pitch = data["rootMotion"]["pitch"] if rooted else 0
    # Do not rename, clear, or replace the user's existing scene objects.
    collection = bpy.data.collections.new("Framecraft Capture")
    scene.collection.children.link(collection)
    first_body = next((f["body"] for f in frames if f.get("body")), None)
    offset = Vector((0, 0, 0))
    if first_body and not rooted:
        foot_z = min(converted(first_body["world"][i]).z for i in (27, 28, 29, 30, 31, 32))
        offset.z = -foot_z + 0.03
    rest_points = body_positions(first_body, offset, pitch, rooted)
    lengths = segment_lengths(frames) if improve_fit else {}
    if improve_fit:
        rest_points = fit_limb_positions(rest_points, lengths)
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
        first_hand_frame = next((f for f in frames if any(h["side"] == side for h in f.get("hands", []))), None)
        if not first_hand_frame:
            continue
        hand_points = positions_for_frame(first_hand_frame, offset, anchors, pitch, rooted)
        # Recenter the first hand pose at the body's rest wrist, not a later wrist location.
        shift = anchors[side] - hand_points[side + "wrist"]
        for key, value in hand_points.items():
            if key.startswith(side + "finger"):
                rest_points[key] = value + shift
        rest_points[side + "wrist"] = anchors[side]
        rest_points[side + "palm"] = rest_points[f"{side}finger9"]
        if not first_body:
            specs.append((side + "Hand", side + "wrist", side + "palm", None))
        for finger, index in (("Thumb", 1), ("Index", 5), ("Middle", 9), ("Ring", 13), ("Pinky", 17)):
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
        rig["capture_root_status"] = "Ground travel is animated on parent Framecraft Root" if rooted else "IN PLACE: root track absent or root import disabled"
        rig["capture_fitting"] = "Median limb lengths; two-bone fitting; measured torso yaw; support-foot anchors" if improve_fit else "Legacy aim-only fitting"
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
    previous_root_time = None
    if rig:
        rest_rot = {b.name: b.matrix_local.to_quaternion() for b in rig.data.bones}
        rest_dir = {b.name: (b.tail_local - b.head_local).normalized() for b in rig.data.bones}
        rest_reference = {name: reference_orientation(rest_dir[name], axis) for name, *_ in specs
                          if (axis := lateral_axis(name, rest_points)) is not None}
    selected = list(range(0, len(frames), frame_step))
    if selected[-1] != len(frames) - 1:
        selected.append(len(frames) - 1)
    context.window_manager.progress_begin(0, len(selected))
    try:
        for progress, index in enumerate(selected):
            frame = frames[index]
            number = start_frame + frame["time"] * metadata["fps"]
            points = positions_for_frame(frame, offset, anchors, pitch, rooted)
            root_sample = frame.get("root") if rooted else None
            if previous_root_time is None or frame["time"] - previous_root_time > .15 or not frame.get("body"):
                foot_locks.clear()
            previous_root_time = frame["time"]
            if improve_fit:
                root_position = Vector(root_location(frame, (0, 0, 0))) if root_sample else None
                contacts = root_sample.get("contacts", {}) if root_sample and root_sample.get("status") == "tracked" else {}
                points = fit_limb_positions(points, lengths, root_position, contacts, foot_locks)
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
                    if improve_fit and name in rest_reference and rest_reference[name] is not None:
                        axis = lateral_axis(name, points)
                        reference = reference_orientation(direction, axis) if axis is not None else None
                        if reference is not None:
                            desired = reference @ rest_reference[name].inverted() @ rest_rot[name]
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

        def execute(self, context):
            try:
                path = Path(self.filepath)
                if path.stat().st_size > 512 * 1024 * 1024:
                    raise ValueError("Capture exceeds the 512 MB import limit.")
                data = json.loads(path.read_text(encoding="utf-8"))
                validate_capture(data)
                import_capture(context, data, self.start_frame, self.frame_step, self.include_face, self.include_root, self.improve_fit)
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
