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
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_location = Vector((0, 0, .9))
            area.spaces.active.region_3d.view_distance = 4
notes = bpy.data.texts.new("Framecraft Capture Notes")
notes.write("Created automatically in a new scene. Existing Blender sessions/scenes were not opened or changed.\n"
            "RTMW3D-X: body/fingers and native face68; no expression scores. Ground travel/scale are estimates.\n"
            "Keep Framecraft Root with Framecraft Rig when exporting. Retargeting to a character is separate.\n"
            + json.dumps(data.get("metadata", {}).get("calibration", {}), indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(destination))
print("FRAMECRAFT_BLEND_READY", flush=True)
