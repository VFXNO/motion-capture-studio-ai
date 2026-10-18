"""Regression: confident bad hand depths must not create/animate giant bones.

Run with --background --factory-startup --python-exit-code 1 --python THIS_FILE.
"""
import copy
import importlib.util
import math
from pathlib import Path

import bpy

spec = importlib.util.spec_from_file_location("framecraft", Path(__file__).parents[1] / "blender_importer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def hand(index):
    world = [{"x": 0, "y": 0, "z": 0, "visibility": .9} for _ in range(21)]
    # Rotate a valid hand and vary observed bone sizes a little. The exported
    # rig must animate directions without growing or separating the phalanges.
    angle = index * .06
    for finger, start in enumerate((1, 5, 9, 13, 17)):
        for segment in range(4):
            x, y = (finger - 2) * .017, (.055 + segment * .025) * (1 + .1 * math.sin(index))
            world[start + segment].update(x=x * math.cos(angle) - y * math.sin(angle),
                                           y=x * math.sin(angle) + y * math.cos(angle))
    if index in (0, 12, 13):
        world[2]["z"] = .8
    return {"side": "Right", "world": world, "score": .99}


frames = [{"time": i / 24, "hands": [hand(i)] if i != 8 else [], "body": None, "face": None} for i in range(24)]
data = {"format": "framecraft-mocap", "version": 1, "metadata": {"width": 640, "height": 480, "fps": 24}, "frames": frames}
before = copy.deepcopy(data)
collection = module.import_capture(bpy.context, data)
assert data == before, "Import must not mutate raw predictions/capture"
rig = next(obj for obj in collection.objects if obj.type == "ARMATURE")
assert len(rig.pose.bones) == 16
assert all(.002 <= bone.length <= .16 for bone in rig.data.bones)
assert all(.002 <= bone.length <= .085 for bone in rig.data.bones if bone.name != "RightHand")
rotations = []
previous = None
for index in range(24):
    for subframe in (0, .5):
        bpy.context.scene.frame_set(index + 1, subframe=subframe if index < 23 else 0)
        wrist = rig.pose.bones["RightHand"].head
        for finger, _ in module.FINGER_CHAINS:
            for segment in (1, 2, 3):
                bone = rig.pose.bones[f"Right{finger}{segment}"]
                assert (bone.tail - wrist).length < .28
                assert abs(bone.length - bone.bone.length) < 1e-5
                if segment > 1:
                    parent = rig.pose.bones[f"Right{finger}{segment - 1}"]
                    assert (bone.head - parent.tail).length < 1e-5
        if subframe == 0:
            matrix = rig.pose.bones["RightThumb1"].matrix.copy()
            if index in (8, 12, 13):
                assert all(abs(a - b) < 1e-5 for arow, brow in zip(matrix, previous) for a, b in zip(arow, brow)), "Bad/missing hand must hold, not explode"
            previous = matrix
            rotations.append(matrix.to_quaternion())
assert any(abs(rotations[1].dot(q)) < .99 for q in rotations[2:]), "Valid hand motion must still animate"

# If the entire hand is invalid, omit its fingers rather than synthesize them.
bad = copy.deepcopy(data)
bad["frames"] = [bad["frames"][0]]
collection = module.import_capture(bpy.context, bad)
assert not any(obj.type == "ARMATURE" for obj in collection.objects)
print("FRAMECRAFT_HAND_FITTING_OK: first-frame outlier, later depth explosions, missing hands, connected fixed-length chains, subframes, no fabricated fingers")
