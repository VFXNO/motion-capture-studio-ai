# RTMW3D-X — no account or SMPL license required

The studio's default **local AI** engine uses RTMW3D-X (133 body, foot, face and
hand points) and YOLOX-M. Their published ONNX model cards and the supporting
code use **Apache 2.0**. No account approval, paid service, license fee, or
SMPL/SMPL-X download is required. They still have license/notice obligations;
see [the sources and notices](backends/rtmw3d/NOTICE.md).

## Normal workflow — automatic

1. Double-click **Launch Studio.bat** and choose your video in the studio.
2. Leave **RTMW3D-X · local AI** selected and click **Extract motion**.
3. Click **Download Blender scene** and open the `.blend`. No importer script,
   setup commands, manual model downloads, Git or Node.js are required.

The first capture runs the setup script automatically, inside `.venv-rtmw3d`,
then performs inference. The browser builds the final capture and the local
server invokes a **fresh background Blender process** to create a new scene.
Your open Blender sessions/files are never opened, cleared or replaced.
Blender is detected automatically; if absent, capture JSON is still available.

Windows and Python 3.12 are required for this engine. The launcher prefers an
installed Python 3.12. Dependencies and about 463 MB of checksum-verified models
download on first use and are reused. Your selected video goes only to the
loopback service on this PC, not to the model hosts. Local job inputs, logs,
predictions and results are private under `.rtmw3d-runs/<random-id>` and are not
served as static files. Cancel stops that job's child process tree. No silent
fallback to MediaPipe occurs. Ordinary app/static files remain allowlisted;
job requests require the local origin/session token and have size limits.

The engine selector also offers **MediaPipe · browser AI** explicitly. Its
capture JSON can be imported manually, as before.

## Advanced: setup only (optional)

In PowerShell, from this project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup-rtmw3d.ps1
```

Creates `.venv-rtmw3d`, downloads checksum-pinned rtmlib source, and downloads about
463 MB of checksum-verified ONNX assets to `.models/rtmw3d`. Does not change
your existing PyTorch/ROCm or GVHMR environment. Uses ONNX Runtime DirectML on
Windows AMD GPUs; some shape/unsupported operators may execute on CPU.

## Advanced: CLI capture and export (optional)

Choose **new** output names; existing files are never replaced:

```powershell
.venv-rtmw3d\Scripts\python.exe tools/rtmw3d.py run "C:\path\video.mp4" "C:\path\Dance.tracking.json" --height 1.7 --fov 60
node tools/export-rtmw3d.mjs "C:\path\Dance.tracking.json" "C:\path\Dance.mocap.json" 10 .25
```

First step saves the actual native predictions. Second step backprojects XY
and depth to estimated meters, applies cleanup, and estimates ground travel.
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

The browser **Extract motion** button uses the selected engine: RTMW3D-X by
default, or MediaPipe when explicitly chosen. GVHMR is not required or used.
