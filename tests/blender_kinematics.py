"""Analytic fitting regression tests. Run in a factory-startup Blender process."""
import importlib.util
import math
from pathlib import Path
import bpy
from mathutils import Vector

spec = importlib.util.spec_from_file_location('framecraft', Path(__file__).parents[1] / 'blender_importer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

for target in (Vector((.2, 0, -.8)), Vector((0, 0, -2)), Vector((0, 0, 0))):
    knee, ankle = module.solve_two_bone(Vector((0, 0, 0)), Vector((0, .2, -.4)), target, .45, .4)
    assert abs(knee.length - .45) < 1e-5
    assert abs((ankle - knee).length - .4) < 1e-5

up = Vector((0, 0, 1))
a = module.reference_orientation(up, Vector((1, 0, 0)))
b = module.reference_orientation(up, Vector((0, 1, 0)))
assert abs(a.rotation_difference(b).angle - math.pi / 2) < 1e-5, 'Measured lateral axis must recover torso yaw'


def body(shift):
    ground = [Vector((0, 0, .6)) for _ in range(33)]
    for sign, indices in ((-1, (11, 13, 15, 23, 25, 27, 29, 31)), (1, (12, 14, 16, 24, 26, 28, 30, 32))):
        values = [(sign*.2, 0, .5), (sign*.4, 0, .25), (sign*.45, 0, .05), (sign*.1, 0, 0),
                  (sign*.1-shift*.5, .1, -.45), (sign*.1-shift, 0, -.9),
                  (sign*.1-shift, -.05, -.98), (sign*.1-shift, .15, -.98)]
        for index, value in zip(indices, values):
            ground[index] = Vector(value)
    world = [{'x': p.x, 'y': -p.z, 'z': p.y, 'visibility': 1} for p in ground]
    return {'world': world, 'landmarks': [{'x': .5, 'y': .5, 'z': 0, 'visibility': 1} for _ in world]}

frames = [{'time': i/24, 'body': body(i*.006), 'hands': [], 'face': None,
           'root': {'position': {'x': i*.006, 'y': 0, 'z': 1}, 'status': 'tracked',
                    'contacts': {'Left': True, 'Right': True}}} for i in range(24)]
data = {'format': 'framecraft-mocap', 'version': 1, 'metadata': {'width': 1280, 'height': 720, 'fps': 24},
        'frames': frames, 'faceEdges': [], 'rootMotion': {'enabled': True, 'method': 'grounded-perspective-v1', 'pitch': 0}}
collection = module.import_capture(bpy.context, data)
rig = next(obj for obj in collection.objects if obj.type == 'ARMATURE')
positions = []
for i in range(24):
    bpy.context.scene.frame_set(i+1)
    foot = rig.matrix_world @ rig.pose.bones['LeftFoot'].tail
    positions.append(foot)
    assert foot.z >= -1e-5, 'Foot tip must not penetrate the floor'
assert max((p-positions[0]).length for p in positions) < .002, 'Support-foot IK must retain its world anchor'
print('FRAMECRAFT_KINEMATICS_OK: fixed lengths, unreachable targets, torso yaw, support-foot lock, ground height')
