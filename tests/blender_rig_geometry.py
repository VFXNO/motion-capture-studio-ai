"""Check a saved armature's geometry, not only its root/face animation.

blender --background capture.blend --python-exit-code 1
  --python tests/blender_rig_geometry.py -- capture.mocap.json
"""
import json
import math
from pathlib import Path
import sys

import bpy

arguments = sys.argv[sys.argv.index("--") + 1:]
data = json.loads(Path(arguments[0]).read_text(encoding="utf8"))
rig = bpy.data.objects["Framecraft Rig"]
height = data["metadata"].get("heightMeters", 1.7)
scale = height / 1.7
finger_bones = [bone for bone in rig.data.bones if any(finger in bone.name for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"))]
oversized = [(bone.name, bone.length) for bone in finger_bones if not .002 * scale <= bone.length <= .085 * scale]
print("FINGER_REST_LENGTHS", json.dumps(sorted([(bone.name, bone.length) for bone in finger_bones], key=lambda item: -item[1])))
assert not oversized, f"Invalid finger rest lengths: {oversized}"
worst_gap, max_radius, limb_gap = 0, 0, 0
preview = []
samples = {round(i * (len(data["frames"]) - 1) / 5) for i in range(6)}
for index, frame in enumerate(data["frames"]):
    number = 1 + frame["time"] * data["metadata"]["fps"]
    bpy.context.scene.frame_set(math.floor(number), subframe=number % 1)
    if index in samples:
        preview.append({"time": frame["time"], "bones": [{"name": bone.name,
            "head": list(rig.matrix_world @ bone.head), "tail": list(rig.matrix_world @ bone.tail)} for bone in rig.pose.bones]})
    for side in ("Left", "Right"):
        for parent, child in (("UpperArm", "LowerArm"), ("UpperLeg", "LowerLeg"), ("LowerLeg", "Foot")):
            a, b = rig.pose.bones[side + parent], rig.pose.bones[side + child]
            limb_gap = max(limb_gap, (a.tail - b.head).length)
            assert abs(a.length - a.bone.length) < 1e-5, "Body limbs must not stretch"
        wrist = rig.pose.bones[side + "Hand"].head
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for segment in (1, 2, 3):
                bone = rig.pose.bones.get(f"{side}{finger}{segment}")
                if not bone:
                    continue
                max_radius = max(max_radius, (bone.head - wrist).length, (bone.tail - wrist).length)
                assert all(math.isfinite(v) for row in bone.matrix for v in row)
                assert abs(bone.length - bone.bone.length) < 1e-5, "Finger bones must not stretch during animation"
                if segment > 1:
                    parent = rig.pose.bones[f"{side}{finger}{segment - 1}"]
                    worst_gap = max(worst_gap, (bone.head - parent.tail).length)
    assert max_radius < .32 * scale, f"Finger escaped the hand at frame {number}: radius {max_radius}m"
    assert worst_gap < .001, f"Disconnected finger chain at frame {number}: gap {worst_gap}m"
    assert limb_gap < .001, f"Disconnected limb at frame {number}: gap {limb_gap}m"
print("SAVED_RIG_GEOMETRY_OK", json.dumps({"frames": len(data["frames"]), "fingerBones": len(finger_bones),
    "maxFingerChainGapM": worst_gap, "maxHandRadiusM": max_radius, "maxLimbChainGapM": limb_gap}))
if len(arguments) > 1:
    with Path(arguments[1]).open("x", encoding="utf8") as report:
        json.dump(preview, report)
