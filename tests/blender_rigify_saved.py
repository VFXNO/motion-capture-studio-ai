"""Verify saved Rigify playback is independent of the source and add-on.

Run in factory-startup Blender: --python this-file -- demo.blend
"""
import sys
import bpy

bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index('--')+1])
rig=bpy.data.objects['Framecraft Rigify Target']
scene=bpy.context.scene
assert scene.render.fps==30 and scene.frame_end==369
action=rig.animation_data.action
assert action and action.get('framecraft_retarget')
expected={}
for frame in (1,68,190,305,366,369):
    scene.frame_set(frame)
    expected[frame]={b.name:b.matrix.copy() for b in rig.pose.bones if b.name.startswith('DEF-') or b.name=='root'}
for name in ('Framecraft Rig','Framecraft Root'):
    bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
worst=0
for frame,matrices in expected.items():
    scene.frame_set(frame)
    for name,matrix in matrices.items():
        actual=rig.pose.bones[name].matrix
        worst=max(worst,max(abs(a-b) for r1,r2 in zip(matrix,actual) for a,b in zip(r1,r2)))
assert worst<1e-6,worst
print('RIGIFY_SAVED_INDEPENDENCE_OK',len(expected),'frames; max matrix difference',worst)
