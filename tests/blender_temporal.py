"""Quaternion continuity, IK bend-plane stability, and hand re-acquisition.

Run in background factory Blender with --python-exit-code 1.
"""
import copy
import importlib.util
import math
from pathlib import Path
import runpy

import bpy
from mathutils import Quaternion, Vector

spec = importlib.util.spec_from_file_location("framecraft", Path(__file__).parents[1] / "blender_importer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
state, previous = {}, None
for i in range(30):
    pole = Vector((.2 * (-1) ** i, 0, -.4))
    elbow, wrist = module.solve_two_bone(Vector(), pole, Vector((0, 0, -.6)), .4, .35, state, math.radians(20))
    assert abs(elbow.length - .4) < 1e-5
    assert abs((wrist - elbow).length - .35) < 1e-5
    if previous is not None:
        assert previous.angle(state["plane"]) <= math.radians(20.01), "IK pole must not flip in one frame"
    previous = state["plane"].copy()

state = {}
for i in range(12):
    endpoint = Vector((.1 * (-1) ** i, .01, -.01))
    before = state.get("direction", Vector()).copy()
    elbow, wrist = module.solve_two_bone(Vector(), Vector((0, .2, -.3)), endpoint,
        .4, .35, state, math.radians(20), math.radians(20))
    if i:
        assert before.angle(state["direction"]) < math.radians(20.01), "Near-shoulder wrist crossings must not flip the entire arm"
    assert abs(elbow.length - .4) < 1e-5 and abs((elbow - wrist).length - .35) < 1e-5

old = Quaternion()
for i in range(40):
    direction = Vector((.01 * math.sin(i), -1, .01 * math.cos(i)))
    rotation = module.transport_orientation(direction, Quaternion(), old)
    assert (rotation @ Vector((0, 1, 0))).angle(direction) < .001
    if i:
        assert abs(old.dot(rotation)) > .999, "Antiparallel rest aim must not cause axial snaps"
    old = rotation

# Use non-default start frame and fractional timestamps to detect unit mix-ups.
target = bpy.data.objects.new("Temporal test", None)
writer = module.CurveWriter(target, "Temporal regression")
times = [i / 30 for i in range(24)]
numbers = [17 + t * 30 for t in times]
mask = [True] * 5 + [False] * 6 + [True] * 13
for i, number in enumerate(numbers):
    angle = 0 if i < 11 else math.radians(165)
    q = Quaternion((0, 0, 1), angle)
    if i % 2:
        q.negate()  # q and -q are the same orientation.
    for axis in range(4):
        writer.add('pose.bones["RightIndex1"].rotation_quaternion', axis, number, q[axis])
    for axis in range(3):
        writer.add('pose.bones["RightIndex1"].location', axis, number, 0)
module.smooth_hand_animation(writer, [("RightIndex1", "", "", None)], times, numbers, {"Right": mask}, .5)
channels = [writer.samples[('pose.bones["RightIndex1"].rotation_quaternion', axis)] for axis in range(4)]
assert channels[0][::2] == numbers
rotations = [Quaternion(tuple(channel[2 * i + 1] for channel in channels)).normalized() for i in range(len(times))]
steps = [math.degrees(2 * math.acos(min(1, abs(a.dot(b))))) for a, b in zip(rotations, rotations[1:])]
assert max(steps) < 36.01
assert steps[10] < 25, "Re-acquisition should ease in, not snap to the new detection"
assert all(step < .01 for step in steps[4:10]), "Missing hands must hold the filtered local pose"
assert abs(rotations[-1].dot(Quaternion((0, 0, 1), math.radians(165)))) > .999

constant = module.CurveWriter(target, "Constant pose")
q = Quaternion((0, 0, 1), math.radians(120))
for number in numbers:
    for axis in range(4):
        constant.add('pose.bones["RightHand"].rotation_quaternion', axis, number, q[axis])
before = copy.deepcopy(constant.samples)
module.smooth_hand_animation(constant, [("RightHand", "", "", None)], times, numbers, {"Right": [True] * len(times)}, 0)
assert constant.samples == before, "0% temporal cleanup must be a no-op"
module.smooth_hand_animation(constant, [("RightHand", "", "", None)], times, numbers, {"Right": [True] * len(times)}, .5)
for i in range(len(times)):
    value = Quaternion(tuple(constant.samples[('pose.bones["RightHand"].rotation_quaternion', axis)][2 * i + 1] for axis in range(4)))
    assert abs(value.dot(q)) > .999, "An initially valid constant pose must not fade from identity"

# Existing floor-lock regression must also pass with the temporal branch enabled.
fixture = runpy.run_path(str(Path(__file__).with_name("blender_kinematics.py")))
data = copy.deepcopy(fixture["data"])
data["metadata"].update(engine="rtmw3d-x", animationSmoothing={"strength": .5})
before = copy.deepcopy(data)
collection = module.import_capture(bpy.context, data, start_frame=7, frame_step=3)
assert data == before
rig = next(obj for obj in collection.objects if obj.type == "ARMATURE")
positions = []
for i in range(0, 24, 3):
    bpy.context.scene.frame_set(7 + i)
    foot = rig.matrix_world @ rig.pose.bones["LeftFoot"].tail
    positions.append(foot)
    assert foot.z >= -1e-5
assert max((p - positions[0]).length for p in positions) < .002
print("FRAMECRAFT_TEMPORAL_OK: stable bend planes, aim transport, quaternion hemisphere, timestamp units, eased reacquisition, fixed geometry and floor locks")
