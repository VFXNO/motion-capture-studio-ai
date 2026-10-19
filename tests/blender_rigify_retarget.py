"""Integration test against a real generated Rigify human, including DEF bones.

Arguments: generated-rig.blend source-capture.blend [new-demo.blend]
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector

args=sys.argv[sys.argv.index('--')+1:]
bpy.ops.wm.open_mainfile(filepath=args[0])
target=bpy.data.objects['rig']
target.name='Framecraft Rigify Target'
target.location.x=2.4
with bpy.data.libraries.load(args[1],link=False) as (available,loaded):
    loaded.objects=[n for n in ('Framecraft Rig','Framecraft Root') if n in available.objects]
for obj in loaded.objects:
    if obj and not obj.users_collection: bpy.context.scene.collection.objects.link(obj)
source=bpy.data.objects['Framecraft Rig']
spec=importlib.util.spec_from_file_location('framecraft_rigify',Path(__file__).parents[1]/'framecraft_rigify.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.register()
settings=bpy.context.scene.framecraft_rigify
settings.source=source; settings.target=target
assert bpy.ops.framecraft_rigify.auto_map()=={'FINISHED'}
rows=[(r.source,r.target,r.role) for r in settings.mappings]
assert len(rows)==49,len(rows)
assert settings.start==1 and settings.end==369
before_source_action=source.animation_data.action
before_parent_action=source.parent.animation_data.action
constraints=sum(len(b.constraints) for b in target.pose.bones)
target.pose.bones['upper_arm_parent.L']['IK_FK']=0.0
target.pose.bones['root'].location.x=.12
target.pose.bones['root'].keyframe_insert('location',frame=1)
old_action=target.animation_data.action
old_action.name='Existing user motion'
bpy.context.scene.frame_set(7,subframe=.25)
start=time.perf_counter()
action=module.retarget(bpy.context,source,target,rows,1,369,1 if len(args)>2 else 30,action_name='Framecraft Dance - Rigify FK')
assert action!=old_action and old_action.name in bpy.data.actions
assert source.animation_data.action==before_source_action and source.parent.animation_data.action==before_parent_action
assert sum(len(b.constraints) for b in target.pose.bones)==constraints
assert bpy.context.scene.frame_current==7 and abs(bpy.context.scene.frame_subframe-.25)<1e-6
scale=json.loads(action['framecraft_retarget'])['scale']
root0=hip0=None
worst=0
for frame in module.frames_between(1,369,1 if len(args)>2 else 30):
    bpy.context.scene.frame_set(frame)
    for side,suffix in (('Left','L'),('Right','R')):
        checks=[('UpperArm','ORG-upper_arm'),('LowerArm','ORG-forearm'),('Hand','ORG-hand'),
                ('UpperLeg','ORG-thigh'),('LowerLeg','ORG-shin'),('Foot','ORG-foot')]
        checks += [(f'{finger}{segment}',f'DEF-{name}.{segment:02d}')
                   for finger,name in (('Thumb','thumb'),('Index','f_index'),('Middle','f_middle'),('Ring','f_ring'),('Pinky','f_pinky'))
                   for segment in range(1,4)]
        for src,dst in checks:
            a=source.pose.bones[side+src]; b=target.pose.bones[dst+'.'+suffix]
            expected=(source.matrix_world.to_3x3() @ (a.tail-a.head)).normalized()
            actual=(target.matrix_world.to_3x3() @ (b.tail-b.head)).normalized()
            error=math.degrees(expected.angle(actual))
            worst=max(worst,error)
            assert error<.15,(frame,side,src,error)
    root=target.matrix_world @ target.pose.bones['root'].head
    hip=source.matrix_world @ source.pose.bones['Hips'].head
    if root0 is None: root0,hip0=root.copy(),hip.copy()
    assert ((root-root0)-(hip-hip0)*scale).length<1e-4
    assert target.pose.bones['upper_arm_parent.L']['IK_FK']==1
assert bpy.ops.framecraft_rigify.previous_action()=={'FINISHED'}
assert target.animation_data.action==old_action
assert target.pose.bones['upper_arm_parent.L']['IK_FK']==0
assert abs(target.pose.bones['root'].location.x-.12)<1e-5
# Failed validation and failed mid-bake evaluation restore the existing action,
# Rigify properties, pose modes and timeline instead of leaving partial work.
try:
    module.retarget(bpy.context,source,target,rows+[rows[0]],1,31)
    raise AssertionError('Duplicate mapping was accepted')
except ValueError:
    pass
assert target.animation_data.action==old_action
parent=bpy.data.objects.new('Animated parent rollback test',None)
bpy.context.scene.collection.objects.link(parent)
target.parent=parent
parent.location.x=0; parent.keyframe_insert('location',frame=1)
parent.location.x=1; parent.keyframe_insert('location',frame=31)
bpy.context.scene.frame_set(7,subframe=.25)
try:
    module.retarget(bpy.context,source,target,rows,1,31,30)
    raise AssertionError('Animated target parent was accepted')
except ValueError as error:
    assert 'animated' in str(error)
assert target.animation_data.action==old_action and target.pose.bones['upper_arm_parent.L']['IK_FK']==0
assert bpy.context.scene.frame_current==7 and abs(bpy.context.scene.frame_subframe-.25)<1e-6
target.parent=None
bpy.data.objects.remove(parent,do_unlink=True)
target.animation_data.action=action
if hasattr(target.animation_data,'action_slot'): target.animation_data.action_slot=action.slots[0]
# Restore baked rotation channels' modes after the explicit previous-action test.
for _,name,_ in rows: target.pose.bones[name].rotation_mode='QUATERNION'
for _,name,_ in rows:
    for collection in target.data.bones[name].collections:
        while collection:
            collection.is_visible=True; collection=collection.parent
target.animation_data.use_nla=False
bpy.context.scene.frame_start=1; bpy.context.scene.frame_end=369
bpy.context.scene.render.fps=30; bpy.context.scene.render.fps_base=1
bpy.context.scene.frame_set(1)
settings.start=1; settings.end=369
settings.action_name='Framecraft Dance - Rigify FK'
for obj in bpy.context.selected_objects: obj.select_set(False)
target.select_set(True); bpy.context.view_layer.objects.active=target
if len(args)>2:
    output=Path(args[2]); assert not output.exists()
    for name in ('Cube','Camera','Light'):
        if name in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                from mathutils import Quaternion
                view=area.spaces.active.region_3d
                view.view_location=Vector((1.2,0,1)); view.view_distance=7
                view.view_rotation=Quaternion((1,0,0),math.pi/2)
                view.view_perspective='ORTHO'
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
print('RIGIFY_RETARGET_OK',json.dumps({'mapped':len(rows),'maxDeformAngleErrorDeg':worst,'travelScale':scale,'seconds':time.perf_counter()-start,'saved':args[2] if len(args)>2 else None}))
