"""Ensure evaluated temporal-rig arm directions preserve the exported 3D pose.

Run against a saved temporal .blend with its matching capture JSON as argument.
This checks raw/refined body17 transport, not ground-truth motion accuracy.
"""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parents[1]))
from blender_importer import ground_coordinates

data = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf8"))
assert data["metadata"]["bodyEstimator"]["rigFitting"] == "model-directions-v1"
rig = bpy.data.objects["Framecraft Rig"]
pitch = data["rootMotion"]["pitch"]
maximum = 0
for frame in data["frames"]:
    if not frame.get("body"):
        continue
    number = 1 + frame["time"] * data["metadata"]["fps"]
    bpy.context.scene.frame_set(math.floor(number), subframe=number % 1)
    for side, indices in (("Left", (11,13,15)), ("Right", (12,14,16))):
        for label, a, b in (("UpperArm",indices[0],indices[1]),("LowerArm",indices[1],indices[2])):
            p = frame["body"]["world"]
            expected = Vector(ground_coordinates(p[b],pitch))-Vector(ground_coordinates(p[a],pitch))
            bone = rig.pose.bones[side+label]
            error = ((bone.tail-bone.head).normalized()-expected.normalized()).length
            maximum = max(maximum,error)
            assert error < .0001, (frame["time"],side,label,error)
assert bpy.data.objects["Framecraft Source View"].data.type == "ORTHO"
print("TEMPORAL_MODEL_DIRECTIONS_OK",json.dumps({"samples":len(data["frames"]),"maxUnitDirectionError":maximum}))
