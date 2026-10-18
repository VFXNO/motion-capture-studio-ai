"""Framewise foot-tip depth flips must not rotate the temporal body joints."""
import math
from pathlib import Path
import sys
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).parents[1]))
from blender_importer import fit_limb_positions

state = {}
lengths = {"Leftthigh":.4,"Leftshin":.4,"Leftfoot":.15}
previous = None
for i in range(24):
    points = {"Lefthip":Vector((0,0,.8)),"Leftknee":Vector((0,0,.4)),"Leftankle":Vector((0,0,0)),
              "Lefttoe":Vector((0,.15*(-1 if i%2 else 1),0))}
    fitted = fit_limb_positions(points,lengths,bend_states=state,dt=1/30,model_body=True)
    direction = fitted["Lefttoe"]-fitted["Leftankle"]
    assert abs(direction.length-.15)<1e-5
    if previous is not None:
        assert math.degrees(direction.angle(previous))<=24.01
    previous = direction.copy()
    assert (fitted["Leftknee"]-points["Leftknee"]).length<.002
    assert (fitted["Leftankle"]-points["Leftankle"]).length<.002
    raw = fit_limb_positions(points,lengths,model_body=True)
    assert (raw["Lefttoe"]-raw["Leftankle"]).normalized().dot(points["Lefttoe"].normalized())>.9999
print("NATIVE_FOOT_CLEANUP_OK: bounded foot tips, fixed lengths, unchanged learned leg, cleanup-off control")
