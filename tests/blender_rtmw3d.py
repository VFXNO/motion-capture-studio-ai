"""Real-video RTMW3D Blender validation; optionally save a NEW .blend file."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

spec = importlib.util.spec_from_file_location("framecraft_importer", Path(__file__).parents[1]/"blender_importer.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
arguments = sys.argv[sys.argv.index("--")+1:]
data = json.loads(Path(arguments[0]).read_text(encoding="utf8"))
assert data["metadata"]["engine"] == "rtmw3d-x"
collection = importer.import_capture(bpy.context, data)
rig = next(obj for obj in collection.objects if obj.type == "ARMATURE")
root = next(obj for obj in collection.objects if obj.name.startswith("Framecraft Root"))
face = next(obj for obj in collection.objects if obj.name.startswith("Framecraft Face"))
assert rig.parent == root and face.parent == root
assert len(face.data.vertices) == 68
assert len(rig.pose.bones) == 49
assert not any(key.startswith("fc_") for key in face.keys()), "RTMW3D cannot fabricate expression scores"
errors, root_errors, rotations, positions = [], [], [], []
for frame in data["frames"]:
    number = 1+frame["time"]*data["metadata"]["fps"]
    bpy.context.scene.frame_set(math.floor(number),subframe=number%1)
    target = Vector(tuple(frame["root"]["position"][axis] for axis in ("x","y","z")))
    root_errors.append((root.matrix_world.translation-target).length)
    positions.append(root.matrix_world.translation.copy())
    if frame.get("face"):
        for i, point in enumerate(frame["face"]["world"]):
            expected = Vector(importer.ground_coordinates(point,data["rootMotion"]["pitch"]))+target
            actual = face.matrix_world @ face.data.vertices[i].co
            errors.append((expected-actual).length)
    rotations.append(rig.pose.bones["LeftIndex1"].rotation_quaternion.copy())
    assert all(math.isfinite(v) for bone in rig.pose.bones for row in bone.matrix for v in row)
assert max(root_errors)<.002
assert max(errors)<.002
assert max((point-positions[0]).length for point in positions)>.02
assert any(abs(rotations[0].dot(value))<.99 for value in rotations[1:]), "Finger animation is static"
print("FRAMECRAFT_RTMW3D_OK",json.dumps({"blender":bpy.app.version_string,"frames":len(data["frames"]),
    "bones":len(rig.pose.bones),"faceVertices":len(face.data.vertices),"maxRootErrorM":max(root_errors),
    "maxFaceWorldErrorM":max(errors),"rootSpansM":importer.root_motion_summary(data)}))
if len(arguments)>1:
    output = Path(arguments[1])
    if output.exists(): raise FileExistsError(output)
    cube=bpy.data.objects.get("Cube")
    if cube: bpy.data.objects.remove(cube,do_unlink=True)
    scene=bpy.context.scene
    scene.frame_end=math.ceil(1+data["frames"][-1]["time"]*data["metadata"]["fps"])
    scene.frame_set(1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=="VIEW_3D":
                area.spaces.active.region_3d.view_location=Vector((0,0,.9))
                area.spaces.active.region_3d.view_distance=4
    notes=bpy.data.texts.new("Framecraft RTMW3D Notes")
    notes.write("RTMW3D-X local ONNX inference; published Apache-2.0 model cards. No GVHMR or SMPL assets used.\n"
                "Body/fingers: Framecraft Rig. Estimated ground travel: parent Framecraft Root; retain it.\n"
                "Face: native 68-point wire mesh, no expression scores.\n"
                "Approximate calibration: height 1.7m, horizontal FOV 60deg, downward tilt 10deg.\n"
                "Root/scale/depth remain estimates. Passing playback tests is not ground-truth accuracy validation.\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print("FRAMECRAFT_RTMW3D_SAVED",output)
