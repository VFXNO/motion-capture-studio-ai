"""Invoked only in a fresh, background Blender process. Never opens user scenes."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

arguments = sys.argv[sys.argv.index("--") + 1:]
source, destination = map(Path, arguments)
if destination.exists():
    raise FileExistsError(f"Not replacing an existing scene: {destination}")
spec = importlib.util.spec_from_file_location("framecraft_importer", Path(__file__).parents[1] / "blender_importer.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
data = json.loads(source.read_text(encoding="utf8"))
collection = importer.import_capture(bpy.context, data)
assert any(obj.type == "ARMATURE" for obj in collection.objects), "No animation rig was created"
cube = bpy.data.objects.get("Cube")
if cube:
    bpy.data.objects.remove(cube, do_unlink=True)
scene = bpy.context.scene
# Export timestamps are rounded; do not add an almost-integer phantom frame.
scene.frame_end = math.ceil(1 + data["frames"][-1]["time"] * data["metadata"]["fps"] - .0001)
scene.frame_set(1)
source_camera = next((obj for obj in collection.objects if obj.type == "CAMERA" and obj.get("capture_review_only")), None)
if source_camera:
    scene.camera = source_camera
    scene.render.resolution_x = data["metadata"]["width"]
    scene.render.resolution_y = data["metadata"]["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_location = Vector((0, 0, .9))
            area.spaces.active.region_3d.view_distance = 4
            if source_camera:
                area.spaces.active.region_3d.view_perspective = "CAMERA"
notes = bpy.data.texts.new("Framecraft Capture Notes")
temporal_body = bool(data["metadata"].get("bodyEstimator"))
notes.write("Created automatically in a new scene. Existing Blender sessions/scenes were not opened or changed.\n"
            + ("RTMPose-X body detections -> MotionAGFormer-B temporal 3D body. RTMW3D fingers/face68/foot details; no expression scores.\n" if temporal_body else "RTMW3D-X: body/fingers and native face68; no expression scores.\n")
            + "Ground travel/scale are estimates, not learned global motion.\n"
            "Hand safety: reject implausible depths; use clip-wide median finger lengths and connected chains. Missing/rejected hand detections hold their last local pose, not recovered hidden motion.\n"
            + ("Importer 0.7.0: temporal body directions retained with fixed limb lengths. No image-ray depth-branch solver; no extra body-target smoothing.\n" if temporal_body else "Legacy capture: image-constrained or 3D-target limb fitting according to metadata.\n")
            +
            "Motion cleanup controls temporal cleanup; 0 disables the cleanup pass, not geometry safeguards. Root/face samples are not modified by the importer.\n"
            "Framecraft Source View is an animated pose-comparison camera, not recovered camera motion. Temporal captures use scaled orthographic projection, consistent with the lifter. Numpad 0 switches the comparison view. No source video is embedded.\n"
            "Keep Framecraft Root with Framecraft Rig when exporting. Retargeting to a character is separate.\n"
            + json.dumps(data.get("metadata", {}).get("calibration", {}), indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(destination))
print("FRAMECRAFT_BLEND_READY", flush=True)
