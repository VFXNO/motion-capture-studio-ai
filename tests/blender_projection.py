"""Compare saved rig joints to native image detections, not ground-truth 3D.

blender --background --factory-startup SCENE --python-exit-code 1
  --python THIS -- CAPTURE TRACKING NEW_REPORT
"""
import json
import math
from pathlib import Path
import statistics
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

arguments = sys.argv[sys.argv.index("--") + 1:]
capture, tracking = [json.loads(Path(p).read_text(encoding="utf8")) for p in arguments[:2]]
calibration = capture["metadata"]["calibration"]
camera = calibration["camera"]
focal, width, height = camera["focal"], camera["width"], camera["height"]
pitch = math.radians(capture.get("rootMotion", {}).get("pitch", 0))
pairs = ((3, 5), (4, 6), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16))
rig = bpy.data.objects["Framecraft Rig"]
specs = {5: ("LeftUpperArm", "head"), 7: ("LeftLowerArm", "head"), 9: ("LeftLowerArm", "tail"),
         6: ("RightUpperArm", "head"), 8: ("RightLowerArm", "head"), 10: ("RightLowerArm", "tail"),
         11: ("LeftUpperLeg", "head"), 13: ("LeftLowerLeg", "head"), 15: ("LeftFoot", "head"),
         12: ("RightUpperLeg", "head"), 14: ("RightLowerLeg", "head"), 16: ("RightFoot", "head")}


def project_native(k, depth):
    center = (k[11][2] + k[12][2]) / 2
    return [((p[0] - width / 2) / focal * (depth + p[2] - center),
             (p[1] - height / 2) / focal * (depth + p[2] - center), depth + p[2] - center) for p in k]


def depth_fit(k):
    def cost(depth):
        positions = project_native(k, depth)
        total, weight = 0, 0
        for (a, b), length in zip(pairs, calibration["lengths"]):
            confidence = min(k[a][3], k[b][3])
            if confidence < .35:
                continue
            error = (math.dist(positions[a], positions[b]) - length) / max(.1, length)
            total += confidence * (error * error if abs(error) < .15 else .3 * abs(error) - .0225)
            weight += confidence
        return total / weight if weight > 1 else math.inf
    lo, hi = max(.5, calibration["medianDepth"] / 3), min(40, calibration["medianDepth"] * 3)
    for _ in range(44):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if cost(a) < cost(b): hi = b
        else: lo = a
    return (lo + hi) / 2


report, camera_errors = [], []
source_camera = next((obj for obj in bpy.data.objects if obj.type == "CAMERA" and obj.get("capture_review_only")), None)
for index, frame in enumerate(capture["frames"]):
    native = min(tracking["frames"], key=lambda f: abs(f["time"] - frame["time"]))
    k = native["keypoints"]
    if k is None or not frame.get("body"):
        continue
    number = 1 + frame["time"] * capture["metadata"]["fps"]
    bpy.context.scene.frame_set(math.floor(number), subframe=number % 1)
    positions = project_native(k, depth_fit(k))
    center = Vector(tuple((positions[11][a] + positions[12][a]) / 2 for a in range(3)))
    if capture["metadata"].get("bodyEstimator"):
        if not frame.get("projection"):
            continue
        center = Vector(tuple(frame["projection"]["cameraCenter"][a] for a in ("x", "y", "z")))
    joints = []
    for joint, (name, attr) in specs.items():
        p = getattr(rig.pose.bones[name], attr)
        cam = Vector((p.x, -math.sin(pitch) * p.y - math.cos(pitch) * p.z,
                      math.cos(pitch) * p.y - math.sin(pitch) * p.z)) + center
        pixel = (width / 2 + focal * cam.x / cam.z, height / 2 + focal * cam.y / cam.z) if cam.z > .05 else (-10000, -10000)
        if capture["metadata"].get("bodyEstimator"):
            scale=frame["projection"]["pixelsPerMeter"]
            pixel=(width/2+scale*cam.x,height/2+scale*cam.y)
        if source_camera:
            view = world_to_camera_view(bpy.context.scene, source_camera, rig.matrix_world @ p)
            camera_errors.append(math.dist((view.x * width, (1 - view.y) * height), pixel))
        joints.append({"index": joint, "native": k[joint][:2], "rig": pixel, "score": k[joint][3],
                       "error": math.dist(pixel, k[joint][:2]), "camera": list(cam)})
    errors = [j["error"] for j in joints if j["score"] >= .5]
    report.append({"index": index, "time": frame["time"], "sourceTime": native.get("sourceTime", frame["time"]),
                   "joints": joints, "meanPixelError": statistics.mean(errors) if errors else None})
errors = [j["error"] for f in report for j in f["joints"] if j["score"] >= .5]
arms = [j["error"] for f in report for j in f["joints"] if j["score"] >= .5 and j["index"] in (7, 8, 9, 10)]
print("VIDEO_REPROJECTION", json.dumps({"frames": len(report), "meanPixelError": statistics.mean(errors),
    "medianPixelError": statistics.median(errors), "armMeanPixelError": statistics.mean(arms),
    "worstFrames": [{"index": f["index"], "time": f["time"], "meanPixelError": f["meanPixelError"]}
                    for f in sorted(report, key=lambda f: -(f["meanPixelError"] or 0))[:6]]}))
if camera_errors:
    assert max(camera_errors) < .05, f"Source camera projection mismatch: {max(camera_errors)} pixels"
    print("SOURCE_CAMERA_PROJECTION_OK", max(camera_errors))
with Path(arguments[2]).open("x", encoding="utf8") as output:
    json.dump(report, output)
