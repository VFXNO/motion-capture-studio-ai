# Framecraft → Rigify retargeting

The **Framecraft Rigify Retarget** add-on bakes a Framecraft capture onto a
generated Rigify human rig's editable **FK controls**. It includes body, hands,
30 finger segments, and scaled ground travel / vertical hip motion.

## Use in Blender

1. Open your capture scene and bring in your character's **generated Rigify rig**
   (not its metarig). The character mesh should already be bound to that rig.
2. In the 3D View, press **N → Framecraft → Rigify Retarget**.
3. Set **Source capture** to `Framecraft Rig` and **Rigify target** to your rig.
4. Click **Auto-map Framecraft to Rigify**. Check the source action's frame range.
5. Click **Bake New Rigify Action**, then play the timeline.

The new action appears in the Action Editor. Rigify's arm/leg `IK_FK` properties
are keyed to **1 (FK)**. FK bone collections are revealed for editing. The
previous action remains available, and **Restore Previous Action / NLA** restores
the prior action, pose settings, collection visibility and NLA state. Undo also
supports the bake operator. Restore applies to the most recent bake on that rig.

## Settings

- **Bone mapping:** editable source bones, target controls and roles; disable a
  row to exclude it. Map control bones, not `DEF-`, `ORG-` or `MCH-` bones.
- **Retarget fingers:** maps individual segments, not guessed finger-curl values.
- **Include ground travel / hip height:** includes source armature-parent travel;
  disabling it makes the retarget in-place.
- **Auto-scale travel:** uses the ratio of target/source leg lengths, retaining
  the target character's limb proportions. **Travel multiplier** adjusts it.
- **Facing offset:** rotates the captured pose and travel around the Z axis.
- **Align first support foot:** offsets the whole animation so the lowest toe on
  its first pose starts on the target's original floor. This is initial alignment,
  not a per-frame foot lock or an IK contact solver.
- **Frame step:** 1 samples every frame; larger values make a lighter action and
  always include the final frame. Playback uses the current scene's FPS.

Framecraft's rest pose can already be mid-dance. This add-on transfers absolute
bone directions with torso/palm axes and continuous roll handling rather than
subtracting that first dance pose as though it were a T-pose. Rigify mechanism
parents are evaluated between control groups before baking local rotations.

No live constraints are added. The baked action can play without the source rig
or this add-on. Existing source actions and Blender files are not rewritten.
Target NLA evaluation is disabled while the new action is active; its previous
state is recorded for the restore button. Face/expression controls are not mapped.

## Install on another Blender installation

Download **Rigify retarget add-on** from the studio, or use
`Framecraft Rigify Retarget.zip`. In Blender:
**Edit → Preferences → Add-ons → Install from Disk**, choose the `.py` or ZIP,
and enable **Framecraft Rigify Retarget**. Rigify is bundled with Blender; enable
it when creating/generating a character rig. No terminal or Text Editor script
is required.

Tested on Blender **5.2.2 LTS**, using the bundled Rigify human metarig and an
actual Framecraft dance capture. The mapper expects Framecraft body names and
standard generated Rigify human FK controls. Use positive uniform object scales
and a stationary target armature object/parent; root motion is baked to its root
control. Different proportions can produce foot sliding, and retargeting cannot
correct errors already present in the capture.
