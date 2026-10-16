"""Synthetic FK/import regression; no licensed assets or inferred model data.

blender --background --factory-startup --python-exit-code 1 --python tests/blender_gvhmr.py
"""
import importlib.util
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Vector, Quaternion

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from gvhmr_fixture import capture, parameters
from backends.gvhmr.motion import PARENTS, NAMES

spec = importlib.util.spec_from_file_location("gvhmr_importer", ROOT / "blender_importer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data = capture()
params, skeletons = parameters()
sentinel = bpy.data.objects.new("User object must survive", None)
bpy.context.scene.collection.objects.link(sentinel)
collection = module.import_capture(bpy.context, data)
rig = next(obj for obj in collection.objects if obj.type == "ARMATURE")
root = rig.parent
conversion = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0))).to_4x4()
origin = Vector(data["metadata"]["horizontalOrigin"])
max_error = 0
for i, skeleton in enumerate(skeletons):
    bpy.context.scene.frame_set(i + 1)
    matrices = []
    aa = [params["global_orient"][i]] + [params["body_pose"][i][j:j+3] for j in range(0, 63, 3)]
    for j, name in enumerate(NAMES):
        # Independent FK in the original Y-up coordinates. Convert only the
        # final world matrices to test the export's handedness and axis angles.
        axis = Vector(aa[j])
        local = (Quaternion(axis.normalized(), axis.length) if axis.length else Quaternion()).to_matrix().to_4x4()
        local.translation = Vector(skeleton[j]) - Vector(skeleton[PARENTS[j]]) if j else Vector(skeleton[0]) + Vector(params["transl"][i])
        world = matrices[PARENTS[j]] @ local if j else local
        matrices.append(world)
        expected = conversion @ world
        expected.translation -= origin
        actual = rig.matrix_world @ rig.pose.bones[name].matrix
        error = (actual.translation - expected.translation).length
        max_error = max(max_error, error)
        assert error < 3e-6, (i, name, error)
        rest_rotation = rig.data.bones[name].matrix_local.to_3x3()
        expected_rotation = (conversion.to_3x3() @ world.to_3x3() @ conversion.to_3x3().transposed()) @ rest_rotation
        difference = actual.to_3x3() - expected_rotation
        assert max(abs(value) for row in difference for value in row) < 3e-6, ("twist/rotation lost", i, name)
    assert (root.matrix_world.translation - Vector(data["frames"][i]["root"])).length < 1e-6
assert bpy.data.objects.get(sentinel.name) is sentinel
assert len(rig.data.bones) == 22
assert not any("finger" in bone.name.lower() for bone in rig.data.bones)

# Disabled travel holds the initial pelvis world position, including hip height.
in_place = module.import_capture(bpy.context, data, include_root=False, frame_step=5)
in_place_rig = next(obj for obj in in_place.objects if obj.type == "ARMATURE")
bpy.context.scene.frame_set(12)
assert (in_place_rig.parent.location - Vector(data["frames"][0]["root"])).length < 1e-6

# Bad captures fail validation BEFORE they add any scene objects.
bad = capture()
bad["frames"][0]["rotations"][0][0] = float("nan")
before = len(bpy.data.objects)
try:
    module.import_capture(bpy.context, bad)
except ValueError:
    pass
else:
    raise AssertionError("Invalid capture was accepted")
assert len(bpy.data.objects) == before
module.register()
module.register()
print("FRAMECRAFT_GVHMR_IMPORT_OK", {"synthetic_frames": len(data["frames"]), "joints": 22,
                                      "max_joint_error_m": max_error, "native_rotations_preserved": True})
