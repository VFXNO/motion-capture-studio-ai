"""Image rays outrank bad model depth; fixed lengths, holds and floor locks remain.

Run with --background --factory-startup --python-exit-code 1 --python THIS.
"""
import copy
import importlib.util
import math
from pathlib import Path
import runpy

import bpy
from mathutils import Vector

spec = importlib.util.spec_from_file_location("framecraft", Path(__file__).parents[1] / "blender_importer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
truth = [Vector(p) for p in ((.15, -.4, 2), (.4, -.4, 2.15), (.4, -.2, 2.3))]
rays = [p / p.z for p in truth]
lengths = [(a - b).length for a, b in zip(truth, truth[1:])]
bad_targets = [truth[0], truth[1] + Vector((0, 0, -.8)), truth[2] + Vector((0, 0, .7))]
solution = module.fit_image_chain(rays, bad_targets, lengths)
assert solution is not None
for p, ray in zip(solution, rays):
    assert (p / p.z - ray).length < 1e-5, "Bad native depth must not override visible 2D joint rays"
for a, b, length in zip(solution, solution[1:], lengths):
    assert abs((a - b).length - length) < 1e-5
assert module.ray_sphere(Vector((0, 0, 1)), Vector((2, 0, 3)), .1) == []

center = Vector((0, 0, 2))
camera = {"rays": [Vector((0, 0, 1)) for _ in range(33)], "scores": [1] * 33, "center": center,
          "to_camera": lambda p: p + center, "to_ground": lambda p: p - center}
for i, ray in zip((11, 13, 15), rays):
    camera["rays"][i] = ray
points = {key: p - center for key, p in zip(("Leftshoulder", "Leftelbow", "Leftwrist"), bad_targets)}
length_dict = {"Leftarm": lengths[0], "Leftforearm": lengths[1]}
states = {}
first = module.fit_limb_positions(points, length_dict, image=camera, image_states=states, bend_states={})
camera["scores"][15] = .1
points["Leftwrist"] += Vector((.5, 0, 0))
held = module.fit_limb_positions(points, length_dict, image=camera, image_states=states, bend_states={})
assert (held["Leftwrist"] - first["Leftwrist"]).length < 1e-5, "Low confidence is a hold, not a new depth-only IK pose"
camera["scores"][15] = 1
camera["rays"][13] = Vector((.16, -.27, 2.1)) / 2.1
camera["rays"][15] = Vector((.3, -.12, 2.15)) / 2.15
resumed = module.fit_limb_positions(points, length_dict, image=camera, image_states=states, bend_states={}, dt=1/30)
assert (resumed["Leftwrist"] - held["Leftwrist"]).length > .001, "The re-acquisition fixture must actually resume tracking"
for a, b in (("Leftshoulder", "Leftelbow"), ("Leftelbow", "Leftwrist")):
    assert (first[b] - first[a]).angle(resumed[b] - resumed[a]) < math.radians(24.01)

fixture = runpy.run_path(str(Path(__file__).with_name("blender_kinematics.py")))
data = copy.deepcopy(fixture["data"])
data["metadata"].update(engine="rtmw3d-x", animationSmoothing={"strength": .5},
                        poseFitting={"method": "image-constrained-limbs-v1"},
                        calibration={"camera": {"width": 1280, "height": 720, "focal": 1000}})
for frame in data["frames"]:
    frame["projection"] = {"cameraCenter": {"x": 0, "y": 0, "z": 3}}
    for point, image in zip(frame["body"]["world"], frame["body"]["landmarks"]):
        image.update(x=.5 + 1000 * point["x"] / (3 + point["z"]) / 1280,
                     y=.5 + 1000 * point["y"] / (3 + point["z"]) / 720)
original = copy.deepcopy(data)
scene_camera = bpy.context.scene.camera
resolution = bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y
collection = module.import_capture(bpy.context, data, start_frame=7)
assert data == original, "Never change original source frames/root/face"
assert bpy.context.scene.camera == scene_camera
assert resolution == (bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
assert any(obj.type == "CAMERA" and obj.get("capture_review_only") for obj in collection.objects)
rig = next(obj for obj in collection.objects if obj.type == "ARMATURE")
positions = []
for i in range(24):
    bpy.context.scene.frame_set(7 + i)
    tip = rig.matrix_world @ rig.pose.bones["LeftFoot"].tail
    positions.append(tip)
    assert tip.z >= -1e-5
assert max((p - positions[0]).length for p in positions) < .002
print("FRAMECRAFT_IMAGE_FITTING_OK: image rays, fixed lengths, low-confidence holds, bounded reacquisition, unchanged data, floor locks, non-destructive comparison camera")
