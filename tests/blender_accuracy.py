"""Measure rig fit, not ground-truth motion accuracy. Run in factory Blender.

--python tests/blender_accuracy.py -- CAPTURE IMPORTER [REPORT.json]
"""
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
data = json.loads(Path(args[0]).read_text(encoding='utf8'))
spec = importlib.util.spec_from_file_location('measured_importer', args[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
collection = module.import_capture(bpy.context, data, include_face=False)
rig = next(obj for obj in collection.objects if obj.type == 'ARMATURE')
root = next(obj for obj in collection.objects if obj.name.startswith('Framecraft Root'))
pitch = data['rootMotion']['pitch']
errors, toes, sliding = [], [], []
previous_contacts = {}
for frame in data['frames']:
    if not frame.get('body') or not frame.get('root'):
        previous_contacts = {}
        continue
    number = 1 + frame['time'] * data['metadata']['fps']
    bpy.context.scene.frame_set(math.floor(number), subframe=number % 1)
    points = module.body_positions(frame['body'], Vector((0, 0, 0)), pitch, True)
    for side in ['Left', 'Right']:
        for bone, key in [('LowerArm', 'elbow'), ('Hand', 'wrist'), ('LowerLeg', 'knee'), ('Foot', 'ankle')]:
            actual = rig.matrix_world @ rig.pose.bones[side + bone].head
            expected = root.matrix_world @ points[side + key]
            errors.append((actual - expected).length)
        foot = rig.pose.bones[side + 'Foot']
        toe = rig.matrix_world @ foot.tail
        toes.append(toe.z)
        contact = frame['root'].get('contacts', {}).get(side, False)
        if contact and side in previous_contacts:
            sliding.append((toe - previous_contacts[side]).xy.length)
        if contact:
            previous_contacts[side] = toe.copy()
        else:
            previous_contacts.pop(side, None)
report = {'blender': bpy.app.version_string, 'samples': len(data['frames']),
          'joint_fit_rms_m': math.sqrt(statistics.mean(e * e for e in errors)),
          'joint_fit_max_m': max(errors), 'min_toe_height_m': min(toes),
          'toe_samples_below_minus_2cm': sum(z < -.02 for z in toes),
          'contact_sliding_mean_m_per_sample': statistics.mean(sliding) if sliding else None}
print('FRAMECRAFT_RIG_METRICS', json.dumps(report))
if len(args) > 2:
    output = Path(args[2])
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2), encoding='utf8')
