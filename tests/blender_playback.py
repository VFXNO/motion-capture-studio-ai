"""Real Blender integration check. Run in a factory-startup background process.

blender --background --factory-startup --python-exit-code 1
  --python tests/blender_playback.py -- CAPTURE.mocap.json
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

spec = importlib.util.spec_from_file_location("framecraft_importer", Path(__file__).parents[1] / "blender_importer.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
path = Path(sys.argv[sys.argv.index("--") + 1])
data = json.loads(path.read_text(encoding="utf8"))
assert data.get("rootMotion", {}).get("enabled"), "Test needs a capture with root travel"
collection = importer.import_capture(bpy.context, data, include_face=True)
rig = next(obj for obj in collection.objects if obj.type == "ARMATURE")
root = next(obj for obj in collection.objects if obj.name.startswith("Framecraft Root"))
face = next((obj for obj in collection.objects if obj.name.startswith("Framecraft Face")), None)
assert rig.parent == root, "Rig must inherit the root trajectory"
if face:
    assert face.parent == root, "Face must travel with the body"
samples = [frame for frame in data["frames"] if frame.get("root")]
positions = []
max_error = 0
for frame in samples:
    time = 1 + frame["time"] * data["metadata"]["fps"]
    bpy.context.scene.frame_set(math.floor(time), subframe=time % 1)
    expected = Vector(tuple(frame["root"]["position"][axis] for axis in ("x", "y", "z")))
    error = (root.matrix_world.translation - expected).length
    max_error = max(max_error, error)
    assert error < 0.002, f"Root did not evaluate at frame {time}: {tuple(root.location)} vs {tuple(expected)}"
    assert (rig.matrix_world.translation - expected).length < 0.002, "Rig world translation is not following root"
    if face:
        assert (face.matrix_world.translation - expected).length < 0.002, "Face root was missing or applied twice"
    positions.append(root.matrix_world.translation.copy())
travel = max((point - positions[0]).length for point in positions)
assert travel > 0.02, "Imported animation never leaves its initial position"
assert rig.animation_data and rig.animation_data.action, "Pose action is missing"
print("FRAMECRAFT_PLAYBACK_OK", json.dumps({"blender": bpy.app.version_string,
    "frames_checked": len(samples), "max_root_error_m": max_error,
    "max_travel_m": travel, "face_tested": bool(face)}))

# Optional ready-to-open scene. This process starts with factory defaults and
# never reads or overwrites an existing .blend scene.
arguments = sys.argv[sys.argv.index("--") + 1:]
if len(arguments) > 1:
    output = Path(arguments[1])
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing scene: {output}")
    cube = bpy.data.objects.get("Cube")
    if cube:
        bpy.data.objects.remove(cube, do_unlink=True)
    scene = bpy.context.scene
    scene.frame_end = math.ceil(1 + data["frames"][-1]["time"] * data["metadata"]["fps"])
    scene.frame_set(1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_location = Vector((0, 0, 0.9))
                area.spaces.active.region_3d.view_distance = 4
    notes = bpy.data.texts.new("Framecraft Root Notes")
    notes.write("Root motion was rebuilt from saved landmarks using FOV 60 degrees and downward tilt 10 degrees.\n"
                "These are estimates, not a calibrated floor.\n"
                "Select Framecraft Rig and play frames 1 through the end.\n"
                "Travel is on its parent Empty, Framecraft Root; retain this parent when moving/exporting the rig.\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print("FRAMECRAFT_SAVED_SCENE", output)
