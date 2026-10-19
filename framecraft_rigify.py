"""Standalone Framecraft -> generated Rigify FK retargeting add-on."""
bl_info = {
    'name': 'Framecraft Rigify Retarget', 'author': 'Framecraft',
    'version': (1, 0, 0), 'blender': (4, 2, 0),
    'location': '3D View > Sidebar > Framecraft', 'category': 'Animation',
    'description': 'Map Framecraft body/fingers to Rigify controls and bake a new action',
}

import json
import math
import statistics

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from mathutils import Matrix, Quaternion, Vector


def default_mapping(source, target):
    rows = [('Hips', 'torso', 'TORSO'), ('Hips', 'hips', 'PELVIS'),
            ('Chest', 'chest', 'CHEST'), ('Neck', 'neck', 'AIM'), ('Head', 'head', 'AIM')]
    for side, suffix in (('Left', 'L'), ('Right', 'R')):
        for src, dst in (('Shoulder','shoulder'), ('UpperArm','upper_arm_fk'), ('LowerArm','forearm_fk'),
                         ('Hand','hand_fk'), ('UpperLeg','thigh_fk'), ('LowerLeg','shin_fk'), ('Foot','foot_fk')):
            rows.append((side+src, dst+'.'+suffix, 'HAND' if src=='Hand' else 'AIM'))
        for finger, name in (('Thumb','thumb'), ('Index','f_index'), ('Middle','f_middle'), ('Ring','f_ring'), ('Pinky','f_pinky')):
            for segment in range(1,4):
                rows.append((f'{side}{finger}{segment}', f'{name}.{segment:02d}.{suffix}', 'FINGER'))
    return [(a,b,role) for a,b,role in rows if a in source.pose.bones and b in target.pose.bones]


def basis(direction, lateral):
    """Anatomical frame: Y along the bone, X across the body/palm."""
    y = direction.normalized()
    x = lateral - y * lateral.dot(y)
    if y.length < .5 or x.length < 1e-6:
        return None
    x.normalize()
    return Matrix((x, y, x.cross(y).normalized())).transposed().to_quaternion()


def frames_between(start, end, step):
    result = list(range(start, end+1, step))
    if result[-1] != end:
        result.append(end)
    return result


def action_curves(action, obj):
    if bpy.app.version < (4,4,0):
        return action.fcurves
    slot = action.slots.new(id_type='OBJECT', name=obj.name)
    obj.animation_data.action_slot = slot
    return action.layers.new('Retarget').strips.new(type='KEYFRAME').channelbag(slot, ensure=True).fcurves


def write_action(target, name, samples):
    action = bpy.data.actions.new(name)
    target.animation_data_create()
    target.animation_data.action = action
    curves = action_curves(action, target)
    try:
        for (path, index), values in samples.items():
            curve = curves.new(data_path=path, index=index)
            curve.keyframe_points.add(len(values)//2)
            curve.keyframe_points.foreach_set('co', values)
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'
            curve.update_autoflags(target)
            curve.update()
    except Exception:
        target.animation_data.action = None
        bpy.data.actions.remove(action)
        raise
    action.use_fake_user = True
    return action


def validate(source, target, rows, start, end, step):
    if not source or not target or source.type != 'ARMATURE' or target.type != 'ARMATURE' or source == target:
        raise ValueError('Choose different source and generated Rigify armatures.')
    if source.mode != 'OBJECT' or target.mode != 'OBJECT':
        if source.mode == 'EDIT' or target.mode == 'EDIT':
            raise ValueError('Leave Edit Mode before baking.')
    if target.library or target.data.library:
        raise ValueError('Make a local copy or editable library override of the target rig.')
    if 'root' not in target.pose.bones or 'torso' not in target.pose.bones or 'rig_id' not in target.data:
        raise ValueError('Target must be the generated Rigify rig, not the metarig.')
    if any(name not in source.pose.bones for name in ('Hips','LeftUpperLeg','RightUpperLeg','LeftUpperArm','RightUpperArm')):
        raise ValueError('Source must use Framecraft Hips, UpperLeg and UpperArm bone names.')
    if any(name not in target.pose.bones for name in ('neck','thigh_fk.L','thigh_fk.R','shin_fk.L','shin_fk.R','upper_arm_fk.L','upper_arm_fk.R')):
        raise ValueError('Use a generated Rigify human rig with standard spine and FK limbs.')
    for obj in (source, target):
        scale = obj.matrix_world.to_scale()
        if min(scale) <= 0 or max(scale)-min(scale) > 1e-4 * max(scale) or obj.matrix_world.determinant() <= 0:
            raise ValueError('Use positive, uniform armature object scale for retargeting.')
    if step < 1 or end < start or len(frames_between(start,end,step)) > 18000:
        raise ValueError('Use an increasing frame range with at most 18,000 baked samples.')
    if not rows:
        raise ValueError('Auto-map the bones first.')
    used = set()
    for src, dst, role in rows:
        if src not in source.pose.bones or dst not in target.pose.bones:
            raise ValueError(f'Mapping references a missing bone: {src} -> {dst}')
        if dst in used or dst == 'root' or dst.startswith(('DEF-','ORG-','MCH-')):
            raise ValueError(f'Map each Rigify control once; invalid target: {dst}')
        if any(not c.mute and c.influence > 0 for c in target.pose.bones[dst].constraints):
            raise ValueError(f'{dst} has active constraints. Map an unconstrained FK control.')
        used.add(dst)
    for suffix in ('L','R'):
        for parent in ('upper_arm_parent','thigh_parent'):
            control = target.pose.bones.get(parent+'.'+suffix)
            if not control or 'IK_FK' not in control:
                raise ValueError(f'Unsupported Rigify limb controls: {parent}.{suffix}')


def snapshot(target):
    ad = target.animation_data
    return {
        'action': ad.action if ad else None,
        'slot': getattr(ad, 'action_slot', None) if ad else None,
        'use_nla': ad.use_nla if ad else True,
        'bones': {b.name: (b.matrix_basis.copy(), b.rotation_mode) for b in target.pose.bones},
        'props': {b.name: {k: b[k] for k in b.keys() if isinstance(b[k], (int,float))} for b in target.pose.bones},
        'visibility': {c.name:c.is_visible for c in target.data.collections_all},
    }


def restore(target, state):
    target.animation_data_create()
    target.animation_data.action = state['action']
    if state['slot'] and hasattr(target.animation_data,'action_slot'):
        target.animation_data.action_slot = state['slot']
    target.animation_data.use_nla = state['use_nla']
    for name, (matrix, mode) in state['bones'].items():
        bone = target.pose.bones[name]
        bone.rotation_mode = mode
        bone.matrix_basis = matrix
    for name, props in state['props'].items():
        for key, value in props.items():
            target.pose.bones[name][key] = value
    for name, visible in state['visibility'].items():
        target.data.collections_all[name].is_visible = visible


def retarget(context, source, target, rows, start, end, step=1, *, fingers=True,
             root_motion=True, auto_scale=True, scale=1.0, yaw=0.0, floor=True,
             action_name='Framecraft Retarget'):
    rows = [r for r in rows if fingers or r[2] != 'FINGER']
    validate(source,target,rows,start,end,step)
    scene = context.scene
    current = (scene.frame_current, scene.frame_subframe)
    state = snapshot(target)
    action = None
    frames = frames_between(start,end,step)
    samples, previous = {}, {}
    def add(path,index,frame,value):
        samples.setdefault((path,index),[]).extend((float(frame),float(value)))
    try:
        target.animation_data_create()
        target.animation_data.action = None
        target.animation_data.use_nla = False
        controls = {dst for src,dst,role in rows} | {'root'}
        controls.update(b.name for b in target.pose.bones if b.name.startswith(('spine_fk','tweak_spine')))
        if fingers:
            controls.update(b.name for b in target.pose.bones if '_master.' in b.name and b.name.startswith(('thumb.','f_index.','f_middle.','f_ring.','f_pinky.')))
        for name in controls:
            target.pose.bones[name].matrix_basis = Matrix.Identity(4)
        for suffix in ('L','R'):
            for parent in ('upper_arm_parent','thigh_parent'):
                b = target.pose.bones[parent+'.'+suffix]
                b['IK_FK'] = 1.0
                if 'FK_limb_follow' in b: b['FK_limb_follow'] = 1.0
            for finger in ('thumb','f_index','f_middle','f_ring','f_pinky'):
                b = target.pose.bones.get(finger+'.01_master.'+suffix)
                if fingers and b and 'finger_curve' in b: b['finger_curve'] = 0.0
        scene.frame_set(start)
        context.view_layer.update()
        target_world = target.matrix_world.copy()
        source_world = source.matrix_world.copy()
        alignment = target_world.to_quaternion() @ Quaternion((0,0,1),yaw) @ source_world.to_quaternion().inverted()
        inverse_world = target_world.inverted()
        source_start = source_world @ source.pose.bones['Hips'].head
        travel_scale = scale
        if auto_scale:
            ratios = []
            for side,suffix in (('Left','L'),('Right','R')):
                a = sum((source_world.to_3x3() @ (source.data.bones[side+b].tail_local-source.data.bones[side+b].head_local)).length for b in ('UpperLeg','LowerLeg'))
                b = sum((target_world.to_3x3() @ (target.data.bones[n+'.'+suffix].tail_local-target.data.bones[n+'.'+suffix].head_local)).length for n in ('thigh_fk','shin_fk'))
                if a > 1e-6: ratios.append(b/a)
            if not ratios: raise ValueError('Cannot estimate leg-length scale; disable Auto Scale and set a manual scale.')
            travel_scale *= statistics.median(ratios)
        rest = {b.name: b.matrix.copy() for b in target.pose.bones}
        foot_tips = [next((n+'.'+s for n in ('toe_fk','toe','foot_fk') if n+'.'+s in target.pose.bones),None) for s in ('L','R')]
        if floor and not all(foot_tips): raise ValueError('Target is missing foot controls for floor alignment.')
        rest_floor = min((target_world @ target.pose.bones[n].tail).z for n in foot_tips) if floor else 0
        target_lateral = target.pose.bones['thigh_fk.L'].head-target.pose.bones['thigh_fk.R'].head
        target_chest_lateral = target.pose.bones['upper_arm_fk.L'].head-target.pose.bones['upper_arm_fk.R'].head
        # Rigify torso/hips/chest controls point horizontally. Their anatomical
        # reference is the trunk, never the display bone's Y axis.
        trunk = target.pose.bones['neck'].head-(target.pose.bones['thigh_fk.L'].head+target.pose.bones['thigh_fk.R'].head)/2
        torso_reference = basis(trunk,target_lateral)
        chest_reference = basis(trunk,target_chest_lateral)
        hand_reference = {}
        for suffix in ('L','R'):
            hand = target.pose.bones.get('hand_fk.'+suffix)
            index = target.pose.bones.get('f_index.01.'+suffix)
            pinky = target.pose.bones.get('f_pinky.01.'+suffix)
            if hand and index and pinky:
                hand_reference[suffix] = basis(hand.tail-hand.head,index.head-pinky.head)
        # Parent-first batches include Rigify mechanism dependencies. Update the
        # dependency graph between depths, then invert the evaluated parent pose.
        levels = {}
        for row in rows:
            depth = len(target.data.bones[row[1]].parent_recursive)
            levels.setdefault(depth,[]).append(row)
        ordered = [levels[k] for k in sorted(levels)]
        root = target.pose.bones['root']
        root_rest = target.data.bones['root'].matrix_local.to_quaternion()
        floor_offset = Vector()
        context.window_manager.progress_begin(0,len(frames))
        for frame_index, frame in enumerate(frames):
            scene.frame_set(frame)
            context.view_layer.update()
            if max(abs(a-b) for row_a,row_b in zip(target.matrix_world,target_world) for a,b in zip(row_a,row_b)) > 1e-5:
                raise ValueError('Target object/parent transform is animated. Retarget onto a stationary rig object.')
            evaluated = source.evaluated_get(context.evaluated_depsgraph_get())
            source_matrix = evaluated.matrix_world.copy()
            points = {b.name:(source_matrix @ b.head,source_matrix @ b.tail) for b in evaluated.pose.bones}
            to_target = inverse_world.to_3x3() @ alignment.to_matrix()
            hip_lateral = points['LeftUpperLeg'][0]-points['RightUpperLeg'][0]
            chest_lateral = points['LeftUpperArm'][0]-points['RightUpperArm'][0]
            displacement = alignment @ (points['Hips'][0]-source_start) * travel_scale if root_motion else Vector()
            root.location = root_rest.inverted() @ (inverse_world.to_3x3() @ (displacement+floor_offset))
            context.view_layer.update()
            for batch in ordered:
                for src,dst,role in batch:
                    head,tail = points[src]
                    direction = to_target @ (tail-head)
                    if direction.length < 1e-6: continue
                    direction.normalize()
                    reference = rest[dst].to_quaternion()
                    old = previous.get(dst)
                    desired = ((old @ Vector((0,1,0))).rotation_difference(direction) @ old if old else
                               (reference @ Vector((0,1,0))).rotation_difference(direction) @ reference)
                    if role in ('TORSO','PELVIS','CHEST'):
                        lateral = hip_lateral if role in ('TORSO','PELVIS') else chest_lateral
                        anatomical = basis(direction,to_target @ lateral)
                        base = torso_reference if role in ('TORSO','PELVIS') else chest_reference
                        if anatomical and base: desired = anatomical @ base.inverted() @ reference
                    elif src in ('Neck','Head'):
                        anatomical = basis(direction,to_target @ chest_lateral)
                        base = basis(reference @ Vector((0,1,0)),target_chest_lateral)
                        if anatomical and base: desired = anatomical @ base.inverted() @ reference
                    elif role == 'HAND':
                        side = 'Left' if src.startswith('Left') else 'Right'
                        suffix = 'L' if side=='Left' else 'R'
                        if side+'Index1' in points and side+'Pinky1' in points:
                            anatomical = basis(direction,to_target @ (points[side+'Index1'][0]-points[side+'Pinky1'][0]))
                            base = hand_reference.get(suffix)
                            if anatomical and base: desired = anatomical @ base.inverted() @ reference
                    pb = target.pose.bones[dst]
                    matrix = desired.to_matrix().to_4x4()
                    matrix.translation = pb.head
                    parent = pb.parent
                    kwargs = {'parent_matrix':parent.matrix.copy(),'parent_matrix_local':parent.bone.matrix_local} if parent else {}
                    local = pb.bone.convert_local_to_pose(matrix,pb.bone.matrix_local,invert=True,**kwargs)
                    pb.rotation_mode = 'QUATERNION'
                    rotation = local.to_quaternion().normalized()
                    old_basis = previous.get('basis:'+dst)
                    if old_basis and rotation.dot(old_basis)<0: rotation.negate()
                    pb.rotation_quaternion = rotation
                    previous[dst] = desired.copy()
                    previous['basis:'+dst] = rotation.copy()
                context.view_layer.update()
            if frame_index==0 and floor:
                actual = min((target_world @ target.pose.bones[n].tail).z for n in foot_tips)
                floor_offset.z = rest_floor-actual
                root.location = root_rest.inverted() @ (inverse_world.to_3x3() @ (displacement+floor_offset))
                context.view_layer.update()
            # Key mapped controls and their neutral spine/master dependencies.
            # Face controls are outside this body's retargeting map.
            for pb in target.pose.bones:
                if pb.name.startswith(('ORG-','DEF-','MCH-','VIS_')): continue
                if pb.name in controls:
                    for path in ('location','scale'):
                        for axis,value in enumerate(getattr(pb,path)): add(pb.path_from_id(path),axis,frame,value)
                    path = 'rotation_quaternion' if pb.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if pb.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
                    for axis,value in enumerate(getattr(pb,path)): add(pb.path_from_id(path),axis,frame,value)
                for key in ('IK_FK','FK_limb_follow','finger_curve'):
                    if key in pb and (key!='finger_curve' or fingers): add(pb.path_from_id('['+json.dumps(key)+']'),0,frame,pb[key])
            context.window_manager.progress_update(frame_index+1)
        action = write_action(target,action_name,samples)
        action['framecraft_retarget'] = json.dumps({'source':source.name,'start':start,'end':end,'step':step,'scale':travel_scale,'rootMotion':root_motion,'method':'absolute-directions-rigify-fk-v1'})
        if state['action']: state['action'].use_fake_user = True
        target['framecraft_previous_action'] = state['action'].name if state['action'] else ''
        target['framecraft_previous_nla'] = state['use_nla']
        target['framecraft_restore_state'] = json.dumps({
            'bones': {name:{'matrix':[list(row) for row in matrix],'mode':mode} for name,(matrix,mode) in state['bones'].items()},
            'props':state['props'], 'visibility':state['visibility'], 'slot':getattr(state['slot'],'identifier',None)})
        target['framecraft_retarget_action'] = action.name
        for name in controls:
            for collection in target.data.bones[name].collections:
                while collection:
                    collection.is_visible=True
                    collection=collection.parent
        scene.frame_set(current[0],subframe=current[1])
        return action
    except Exception:
        restore(target,state)
        if action: bpy.data.actions.remove(action)
        scene.frame_set(current[0],subframe=current[1])
        raise
    finally:
        context.window_manager.progress_end()


def armature_poll(self,obj):
    return obj.type=='ARMATURE'


class FRAMECRAFT_RIGIFY_Map(bpy.types.PropertyGroup):
    enabled: BoolProperty(default=True)
    source: StringProperty(name='Source bone')
    target: StringProperty(name='Rigify control')
    role: EnumProperty(name='Role',items=[(x,x.title(),'') for x in ('TORSO','PELVIS','CHEST','AIM','HAND','FINGER')],default='AIM')


class FRAMECRAFT_RIGIFY_Settings(bpy.types.PropertyGroup):
    source: PointerProperty(name='Source capture',type=bpy.types.Object,poll=armature_poll)
    target: PointerProperty(name='Rigify target',type=bpy.types.Object,poll=armature_poll)
    mappings: CollectionProperty(type=FRAMECRAFT_RIGIFY_Map)
    start: IntProperty(name='Start',default=1)
    end: IntProperty(name='End',default=250)
    step: IntProperty(name='Frame step',default=1,min=1,max=120)
    fingers: BoolProperty(name='Retarget fingers',default=True)
    root_motion: BoolProperty(name='Include ground travel / hip height',default=True)
    auto_scale: BoolProperty(name='Auto-scale travel to target legs',default=True)
    scale: FloatProperty(name='Travel multiplier',default=1,min=.001,max=100)
    yaw: FloatProperty(name='Facing offset',subtype='ANGLE',default=0)
    floor: BoolProperty(name='Align first support foot to target floor',default=True)
    action_name: StringProperty(name='New action',default='Framecraft Retarget')
    show_mapping: BoolProperty(name='Bone mapping',default=False)


class FRAMECRAFT_RIGIFY_OT_map(bpy.types.Operator):
    bl_idname='framecraft_rigify.auto_map'
    bl_label='Auto-map Framecraft to Rigify'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene.framecraft_rigify
        if not s.source or not s.target or s.source==s.target:
            self.report({'ERROR'},'Choose different source and target armatures.'); return {'CANCELLED'}
        s.mappings.clear()
        for a,b,role in default_mapping(s.source,s.target):
            row=s.mappings.add(); row.source=a; row.target=b; row.role=role
        action=s.source.animation_data.action if s.source.animation_data else None
        if action:
            s.start=math.floor(action.frame_range[0]+.0001); s.end=math.ceil(action.frame_range[1]-.0001)
        else:
            s.start=context.scene.frame_start; s.end=context.scene.frame_end
        self.report({'INFO'},f'Mapped {len(s.mappings)} controls. Review the range, then Bake New Action.')
        return {'FINISHED'}


class FRAMECRAFT_RIGIFY_OT_add(bpy.types.Operator):
    bl_idname='framecraft_rigify.add_mapping'
    bl_label='Add Bone Mapping'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        context.scene.framecraft_rigify.mappings.add(); return {'FINISHED'}


class FRAMECRAFT_RIGIFY_OT_bake(bpy.types.Operator):
    bl_idname='framecraft_rigify.bake'
    bl_label='Bake New Rigify Action'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s=context.scene.framecraft_rigify
        rows=[(r.source,r.target,r.role) for r in s.mappings if r.enabled]
        try:
            action=retarget(context,s.source,s.target,rows,s.start,s.end,s.step,fingers=s.fingers,
                            root_motion=s.root_motion,auto_scale=s.auto_scale,scale=s.scale,yaw=s.yaw,
                            floor=s.floor,action_name=s.action_name)
        except Exception as error:
            self.report({'ERROR'},str(error)); return {'CANCELLED'}
        self.report({'INFO'},f'Baked {action.name} on {s.target.name}. Rigify limbs use FK.')
        return {'FINISHED'}


class FRAMECRAFT_RIGIFY_OT_previous(bpy.types.Operator):
    bl_idname='framecraft_rigify.previous_action'
    bl_label='Restore Previous Action / NLA'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        target=context.scene.framecraft_rigify.target
        if not target or 'framecraft_retarget_action' not in target: return {'CANCELLED'}
        name=target.get('framecraft_previous_action','')
        if name and name not in bpy.data.actions:
            self.report({'ERROR'},'Previous action was renamed or removed. Select it in the Action Editor.'); return {'CANCELLED'}
        saved=json.loads(target.get('framecraft_restore_state','{}'))
        for bone,state in saved.get('bones',{}).items():
            if bone in target.pose.bones:
                target.pose.bones[bone].rotation_mode=state['mode']
                target.pose.bones[bone].matrix_basis=Matrix(state['matrix'])
        for bone,props in saved.get('props',{}).items():
            if bone in target.pose.bones:
                for key,value in props.items(): target.pose.bones[bone][key]=value
        for collection_name,visible in saved.get('visibility',{}).items():
            if collection_name in target.data.collections_all: target.data.collections_all[collection_name].is_visible=visible
        target.animation_data.action=bpy.data.actions.get(name) if name else None
        if name and saved.get('slot') and hasattr(target.animation_data,'action_slot'):
            slot=next((s for s in target.animation_data.action.slots if s.identifier==saved['slot']),None)
            if slot: target.animation_data.action_slot=slot
        target.animation_data.use_nla=bool(target.get('framecraft_previous_nla',True))
        context.scene.frame_set(context.scene.frame_current)
        return {'FINISHED'}


class FRAMECRAFT_RIGIFY_PT_panel(bpy.types.Panel):
    bl_label='Rigify Retarget'
    bl_idname='FRAMECRAFT_RIGIFY_PT_panel'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_category='Framecraft'
    def draw(self,context):
        layout=self.layout; s=context.scene.framecraft_rigify
        layout.prop(s,'source'); layout.prop(s,'target')
        layout.operator('framecraft_rigify.auto_map',icon='BONE_DATA')
        row=layout.row(align=True); row.prop(s,'start'); row.prop(s,'end'); row.prop(s,'step')
        layout.prop(s,'fingers'); layout.prop(s,'root_motion'); layout.prop(s,'auto_scale')
        layout.prop(s,'scale'); layout.prop(s,'yaw'); layout.prop(s,'floor'); layout.prop(s,'action_name')
        layout.operator('framecraft_rigify.bake',icon='ACTION')
        layout.operator('framecraft_rigify.previous_action',icon='LOOP_BACK')
        layout.label(text='Generated Rigify FK controls; body + fingers.')
        layout.prop(s,'show_mapping',icon='TRIA_DOWN' if s.show_mapping else 'TRIA_RIGHT',emboss=False)
        if s.show_mapping:
            for item in s.mappings:
                box=layout.box(); box.prop(item,'enabled'); box.prop(item,'role')
                if s.source: box.prop_search(item,'source',s.source.pose,'bones')
                if s.target: box.prop_search(item,'target',s.target.pose,'bones')
            layout.operator('framecraft_rigify.add_mapping',icon='ADD')


CLASSES=(FRAMECRAFT_RIGIFY_Map,FRAMECRAFT_RIGIFY_Settings,FRAMECRAFT_RIGIFY_OT_map,
         FRAMECRAFT_RIGIFY_OT_add,FRAMECRAFT_RIGIFY_OT_bake,FRAMECRAFT_RIGIFY_OT_previous,FRAMECRAFT_RIGIFY_PT_panel)


def register():
    for cls in CLASSES: bpy.utils.register_class(cls)
    bpy.types.Scene.framecraft_rigify=PointerProperty(type=FRAMECRAFT_RIGIFY_Settings)


def unregister():
    del bpy.types.Scene.framecraft_rigify
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)


if __name__=='__main__': register()
