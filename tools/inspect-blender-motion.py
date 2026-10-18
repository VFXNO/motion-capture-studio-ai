"""Measure a saved rig's frame steps (not ground-truth capture accuracy).

blender --background --factory-startup SCENE --python THIS -- CAPTURE [NEW_REPORT]
"""
import json
import math
from pathlib import Path
import statistics
import sys
import time

import bpy

arguments = sys.argv[sys.argv.index("--") + 1:]
capture = json.loads(Path(arguments[0]).read_text(encoding="utf8"))
rig = bpy.data.objects["Framecraft Rig"]
previous, steps, evaluation = {}, {}, []
for index, sample in enumerate(capture["frames"]):
    frame = 1 + sample["time"] * capture["metadata"]["fps"]
    start = time.perf_counter()
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    evaluation.append((time.perf_counter() - start) * 1000)
    for bone in rig.pose.bones:
        rotation = bone.rotation_quaternion.normalized()
        head, tail = bone.head.copy(), bone.tail.copy()
        if bone.name in previous:
            old_rotation, old_head, old_tail = previous[bone.name]
            angle = math.degrees(2 * math.acos(min(1, abs(rotation.dot(old_rotation)))))
            steps.setdefault(bone.name, []).append({"index": index, "time": sample["time"],
                "angle": angle, "step": max((head - old_head).length, (tail - old_tail).length)})
        previous[bone.name] = rotation, head, tail


def percentile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


summary = {name: {"peakAngleDeg": max(s["angle"] for s in values),
    "p95AngleDeg": percentile([s["angle"] for s in values], .95),
    "peakJointStepM": max(s["step"] for s in values),
    "worstFrames": sorted(values, key=lambda s: s["angle"], reverse=True)[:3]}
    for name, values in steps.items()}
groups = {group: [s for name, values in steps.items() if ("Hand" in name or any(f in name for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"))) == (group == "hands") for s in values]
          for group in ("hands", "body")}
report = {"samples": len(capture["frames"]), "importer": rig.get("capture_importer_version"),
    "evaluationMs": {"median": statistics.median(evaluation), "p95": percentile(evaluation, .95)},
    "groups": {name: {"p95AngleDeg": percentile([s["angle"] for s in values], .95),
                       "peakAngleDeg": max(s["angle"] for s in values),
                       "over45DegSteps": sum(s["angle"] > 45 for s in values)} for name, values in groups.items()},
    "bones": summary}
print("MOTION_CONTINUITY", json.dumps({k: v for k, v in report.items() if k != "bones"}))
print("WORST_BONES", json.dumps([{ "name": name, "peakAngleDeg": values["peakAngleDeg"],
    "p95AngleDeg": values["p95AngleDeg"], "peakJointStepM": values["peakJointStepM"]}
    for name, values in sorted(summary.items(), key=lambda item: -item[1]["peakAngleDeg"])[:5]]))
if len(arguments) > 1:
    with Path(arguments[1]).open("x", encoding="utf8") as output:
        json.dump(report, output, indent=2)
