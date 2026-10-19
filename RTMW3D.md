# Local temporal AI — no account or SMPL download required

The studio's default **local AI** engine uses **RTMPose-X** for body image
detections, **MotionAGFormer-B** for temporal 3D body estimation, **RTMW3D-X**
for hands/face/foot detail, and YOLOX-M for performer detection. Upstream projects
use **Apache 2.0**; see the exact model provenance and conversion verification
in the notices. No account approval, paid service, license fee, or
SMPL/SMPL-X download is required. They still have license/notice obligations;
see [the sources and notices](backends/rtmw3d/NOTICE.md).

## Normal workflow — automatic

1. Double-click **Launch Studio.bat** and choose your video in the studio.
2. Leave **MotionAGFormer + RTMW3D · local AI** selected and click **Extract motion**.
3. Click **Download Blender scene** and open the `.blend`. No importer script,
   setup commands, manual model downloads, Git or Node.js are required.

The first capture runs the setup script automatically, inside `.venv-rtmw3d`,
then performs inference. The browser builds the final capture and the local
server invokes a **fresh background Blender process** to create a new scene.
Your open Blender sessions/files are never opened, cleared or replaced.
Blender is detected automatically; if absent, capture JSON is still available.

Windows and Python 3.12 are required for this engine. The launcher prefers an
installed Python 3.12. Dependencies and about 695 MB of checksum-verified models
download on first use and are reused. Your selected video goes only to the
loopback service on this PC, not to the model hosts. Local job inputs, logs,
predictions and results are private under `.rtmw3d-runs/<random-id>` and are not
served as static files. Cancel stops that job's child process tree. No silent
fallback to MediaPipe occurs. Ordinary app/static files remain allowlisted;
job requests require the local origin/session token and have size limits.

The engine selector also offers **MediaPipe · browser AI** explicitly. Its
capture JSON can be imported manually, as before.

## Rebuilt body estimation (importer 0.7.1)

The default now runs a **trained temporal neural network**, rather than making
independent 3D body estimates and repairing them with the image-ray fitter.
The dedicated RTMPose-X detector supplies body17 XY/confidence. MotionAGFormer-B
uses 243-frame windows, horizontal-flip inference and overlapping 81-frame
strides, resampled at 30 Hz with original output timestamps preserved. Missing
body sequences split context; they are not silently replaced by the old model.
Portrait videos are square-letterboxed before normalized-coordinate inference.

The body network runs explicitly on CPU (about 1 second per window on the test
PC). Its GCN graph does not initialize with DirectML. Image detectors still run
on DirectML by default. All setup is automatic; no base Python/PyTorch changes,
SMPL assets, Torch installs or GPU-specific CUDA wheels are needed.

After inference, observed-arm refinement corrects neural XY errors using
RTMPose-X image vectors, detector confidence/agreement, and a robust camera scale
estimated from the torso/legs. Neural depth guides a sequence-wide depth-sign
selection; continuous depth transitions avoid instantaneous hemisphere flips.
A stronger dedicated-body detection can outweigh an uncertain RTMW3D joint.
This is always part of estimation; the Motion cleanup slider controls auxiliary
hand/foot cleanup, not this image/depth fitting stage.

The neural torso/legs are retained. Blender follows the resulting body17
directions and uses clip-median limb lengths.
The temporal branch bypasses the previous image-ray depth-branch solver and
additional importer body-target smoothing.
Foot-tip directions still come from RTMW3D; with Motion cleanup enabled their
angular changes are bounded separately, without filtering the learned body17.
Source comparison uses **scaled
orthographic projection**, consistent with the model's lifting assumptions.
Root travel is estimated from projected size/FOV and foot contacts; it is not
a learned world trajectory. Keep the root parent when exporting.

`.tracking.json` retains original RTMW3D predictions, separate `body2d` RTMPose
detections, raw `temporalBody` model output, and separate `refinedBody` results.
`metadata.bodyEstimator.refinement` identifies the observed-arm algorithm.
Capture JSON identifies
`metadata.bodyEstimator` and carries the 17-joint body plus compatibility body33.
Hands/face/foot detail still come from RTMW3D; this body network does not recover
hidden fingers or facial expression scores. Human3.6M-trained lifting can still
misestimate unusual dance poses and occlusions. Playback, continuity and
conversion tests are not proof of ground-truth 3D accuracy.

Refresh the studio and extract again to use this pipeline. Existing scenes do
not change. For advanced reuse, `tools/rtmw3d.py lift OLD.tracking.json NEW.tracking.json`
adds temporal inference; old tracks without `body2d` use their recorded RTMW3D
XY instead of the new dedicated detector. `run --body-model legacy` explicitly
selects the earlier framewise model for comparison.

### Observed-arm correction (2026-10-19)

`Documents/Framecraft Dance Arm Corrected.blend` and matching capture JSON use
the new arm-refinement stage. `Framecraft Dance Arm Corrected Comparison.mp4`
shows source / previous temporal rig / corrected rig at matching timestamps.
The clearly visible raised wrist around 2.23s now points upward with the source
instead of folding toward the head; the transition around 10.97s also follows
the detected arm. Review still cannot establish hidden-arm depth at 10.13s.

For this 369-sample scene, body orientation steps over 45 degrees fell from 3
to 0; hand steps over 45 degrees remain 0. Mean arm reprojection discrepancy
against RTMW3D image detections fell from 18.76 to 12.55 pixels. These are
diagnostics, not ground-truth accuracy: the obscured arm at 10.13s is a case
where RTMW3D and RTMPose-X disagree. Raw predictions remain byte-equivalent as
JSON values, and non-arm body17 joints are unchanged by the new refinement.
48 Python tests, 34 JavaScript tests, real browser capture/export and final
saved-scene geometry, root/face transport and arm-direction checks passed.

### Earlier temporal-body delivery (2026-10-18)

Delivered `Documents/Framecraft Dance Temporal AI.blend`, matching capture JSON,
and `Framecraft Dance Temporal AI Comparison.mp4` (source beside evaluated rig
projection). The 369-sample scene retains model arm directions, connected fixed
limbs/fingers and the exported root/face animation. Native detector predictions
remain separate and unchanged. Body coordinates match scaled model output within
JSON rounding. Tests include 34 JavaScript tests, 43 Python tests, browser workflow
tests with real inference/export, and saved-scene geometry/direction checks.

This is **not a verified solution to all pose errors**. Sampled overlays still
show arm mismatches around 2.3s, 10.13s, 10.97s and 12.17s. Three body orientation
steps exceed 45 degrees, concentrated in the left arm; native foot-tip cleanup
does not resolve those learned depth changes. Reprojection and continuity scores
are diagnostics, not ground-truth 3D accuracy. Ground travel remains approximate.

## Legacy video-constrained fitting (0.6.0 captures)

Older image-constrained captures retain a camera center for each sample. Blender fits
fixed-length arms/legs to **visible joint image rays**, rather than letting a
bad model depth move a correctly detected elbow/wrist/knee away from the video.
Weak model-depth and temporal priors choose among ambiguous depth branches.
Conflicting or low-confidence arm detections hold the last shoulder-relative
pose; re-acquisition has bounded angular steps. Feet still obey support/floor
constraints after fitting. Exact image alignment can be sacrificed temporarily
when rays conflict with fixed lengths, a floor lock or temporal safety limits.
This is not learned 3D reconstruction or a guarantee that hidden depth is right.

The new scene opens in **Framecraft Source View** (Numpad 0 toggles camera view).
Compare the same timestamp and camera angle, not an arbitrary perspective.
This animated review camera compensates approximate root stabilization; it is
**not recovered camera motion** and does not change the rig/root animation.
No source video is embedded. Manual import adds the review camera without
replacing an existing scene camera or changing its render settings. Older JSON
without projection metadata keeps the previous 3D-target fitting; regenerate
from saved native tracking or extract again to get the new temporal body model.

Confident RTMW3D hand predictions can still contain implausible depth, especially
when hands are occluded. Export now rejects implausible hand geometry before
cleanup. Blender selects a reliable rest sample, estimates palm/finger lengths
across the clip, and keeps finger chains connected with fixed lengths. The
importer applies these safeguards to older capture JSON too. Rejected or missing
hands hold their last local pose; hidden finger motion is not reconstructed.
If no reliable hand is available anywhere, its finger bones are omitted.
Raw `.tracking.json` predictions remain unchanged. This fixes rig distortion,
not the underlying monocular model's accuracy limits.

Download a newly generated `.blend` or re-import JSON with the updated importer;
previously exported Blender scenes do not update themselves.

To reduce snapping, the importer now smooths body targets with a centered,
timestamp-based filter **before** fixed-length IK. Implausible one-frame speeds,
collapsed torso axes and elbow/knee pole flips are stabilized. Bone aiming uses
continuous minimum-twist transport instead of resolving unconstrained twist
from the rest pose independently every frame. Hand rotations use quaternion
filtering, bounded angular steps, and an approximately 0.14–0.21 second ease-in
when tracking resumes. Missing hands still hold; gaps are not labeled as fresh
detections, and long gaps are not filled with invented motion.

The studio's **Motion cleanup** setting controls this temporal pass too; 0%
disables it while preserving hand geometry safeguards. Manual import also has
**Smooth RTMW3D pose timing and hand re-acquisition**. Existing captures without
a saved setting use 35%. Limb lengths and finger chains stay fixed, support
foot constraints run after cleanup, and root/face samples are left unchanged.
Smoothing can soften very fast motion; it does not recover reliable depth or
prove greater accuracy. GVHMR/MediaPipe behavior is unchanged.

## Advanced: setup only (optional)

In PowerShell, from this project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup-rtmw3d.ps1
```

Creates `.venv-rtmw3d`, downloads checksum-pinned rtmlib source, and downloads about
695 MB of checksum-verified ONNX assets to `.models/rtmw3d`. Does not change
your existing PyTorch/ROCm or GVHMR environment. Uses ONNX Runtime DirectML on
Windows AMD GPUs; some shape/unsupported operators may execute on CPU.

## Advanced: CLI capture and export (optional)

Choose **new** output names; existing files are never replaced:

```powershell
.venv-rtmw3d\Scripts\python.exe tools/rtmw3d.py run "C:\path\video.mp4" "C:\path\Dance.tracking.json" --height 1.7 --fov 60
node tools/export-rtmw3d.mjs "C:\path\Dance.tracking.json" "C:\path\Dance.mocap.json" 10 .25
```

First step saves native wholebody, dedicated body XY and temporal 3D predictions.
Second step scales the temporal body, attaches native details, applies auxiliary
hand/face cleanup, and estimates ground travel. Learned body17 and body33 joints
are kept consistent and bypass additional body smoothing in both export and import.
The final two parameters are camera downward tilt (degrees) and cleanup strength
(0–0.85). Height/FOV/tilt defaults are estimates; adjust to the real performer
and camera for better scale. Node.js is required for export.

`--device cpu` explicitly uses CPU instead of DirectML. `--frames 3` runs a
short inference smoke test only. Use one clearly visible performer, a fixed
camera and a flat floor; clips must be at most 60 seconds. Sampling follows
video presentation timestamps (including variable-frame-rate sources), at
24 fps by default (`--fps 12`, `24`, or `30`). No video is uploaded.

## Manual Blender import (optional)

Run the updated `blender_importer.py` in Blender's Text Editor, then use
**File → Import → Framecraft Capture** on the final `.mocap.json`.

- Body and fingers animate on `Framecraft Rig`.
- Estimated ground travel is on its parent `Framecraft Root`; retain it.
- Native **68-point** facial wire mesh, not a 478-point MediaPipe face and
  **no expression/blendshape scores**. Not a ready-made character face rig.
- Body joints are explicitly mapped to the existing body33 transport for
  Blender compatibility. This workflow does **not** run MediaPipe inference.

## Important accuracy limits

RTMW3D outputs **image XY and relative depth**, not absolute metric XYZ or
learned world travel. We backproject using estimated FOV and actor height,
fit clip-wide segment lengths, and reuse the fixed-camera ground estimator.
Head/sole allowance (14 cm), body scale, floor, contacts and travel remain
approximate. No SMPL fitting, SLAM, hidden-finger recovery or native bone twist.
Fast/occluded motion can still fail. Successful inference/Blender playback
does not prove improved motion accuracy; compare visually to your video.

The browser **Extract motion** button uses the selected engine: the local
MotionAGFormer + RTMW3D pipeline by default, or MediaPipe when explicitly chosen.
GVHMR is not required or used.
