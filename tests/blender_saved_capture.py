"""Verify an already-saved automatic .blend, not a newly re-imported rig.

blender --background scene.blend --python-exit-code 1
  --python tests/blender_saved_capture.py -- scene.mocap.json
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

spec=importlib.util.spec_from_file_location("framecraft_importer",Path(__file__).parents[1]/"blender_importer.py")
importer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
data=json.loads(Path(sys.argv[sys.argv.index("--")+1]).read_text(encoding="utf8"))
root=bpy.data.objects["Framecraft Root"]
rig=bpy.data.objects["Framecraft Rig"]
face=bpy.data.objects["Framecraft Face"]
assert rig.parent==root and face.parent==root
assert rig["capture_engine"]=="rtmw3d-x"
assert len(rig.pose.bones)==49 and len(face.data.vertices)==68
assert rig.animation_data.action and face.data.animation_data.action and root.animation_data.action
errors,face_errors,rotations=[],[],[]
for frame in data["frames"]:
    number=1+frame["time"]*data["metadata"]["fps"]
    bpy.context.scene.frame_set(math.floor(number),subframe=number%1)
    target=Vector(tuple(frame["root"]["position"][axis] for axis in ("x","y","z")))
    errors.append((root.matrix_world.translation-target).length)
    for index,point in enumerate(frame["face"]["world"]):
        expected=Vector(importer.ground_coordinates(point,data["rootMotion"]["pitch"]))+target
        actual=face.matrix_world@face.data.vertices[index].co
        face_errors.append((actual-expected).length)
    rotations.append(rig.pose.bones["LeftIndex1"].rotation_quaternion.copy())
assert max(errors)<.002 and max(face_errors)<.002
assert any(abs(rotations[0].dot(value))<.99 for value in rotations[1:])
assert bpy.context.scene.frame_end==math.ceil(1+data["frames"][-1]["time"]*data["metadata"]["fps"]-.0001)
print("SAVED_AUTOMATIC_BLEND_OK",json.dumps({"samples":len(data["frames"]),"maxRootErrorM":max(errors),
    "maxFaceErrorM":max(face_errors),"blender":bpy.app.version_string}))
