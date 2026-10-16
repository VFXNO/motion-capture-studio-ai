# GVHMR body + world-root backend

This is a separate **local Python/GPU engine**. The studio's **Extract motion** button runs RTMW3D-X automatically by default, or MediaPipe when selected—not GVHMR. Use the CLI below only for the optional licensed GVHMR workflow.

## Current verification and limits

- The isolated Windows environment was tested with the RX 9070 XT (15.9 GiB), ROCm PyTorch, GPU matrix/rotation/NMS operations, and headless GVHMR inference imports/configuration.
- A random-weight GVHMR temporal-network forward pass also ran on the GPU; timestamp resampling from 24 to 30 fps was tested without changing video duration. These are compatibility checks, not trained-model inference or accuracy tests.
- Native Blender rotation/root transport is regression-tested with **synthetic** motion, including body twist, changing shape offsets, and root travel applied once.
- **Actual pretrained GVHMR inference has not been verified on this AMD machine.** Missing checkpoints and the licensed SMPL-X asset block that test. GPU compatibility and improved motion quality cannot be promised until that run succeeds.
- GVHMR estimates **body + world trajectory only**: 22 body-joint rotations, no articulated fingers, no facial expressions. Browser hand/face captures are not automatically merged into this differently oriented rig.
- GVHMR is educational/research/non-profit software, **not commercially licensed by this setup**. Read the [upstream license](https://github.com/zju3dv/GVHMR/blob/main/LICENSE), [installation/asset terms](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md), and each model's terms. Commercial use requires appropriate permission. We do not accept licenses or log into accounts for you.

## 1. Environment (already prepared on this machine)

From PowerShell in `C:\Users\bsher\motion-capture-studio`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup-gvhmr.ps1
```

The script creates `.venv-gvhmr` with read-only access to the existing GPU PyTorch/torchvision installation. Other packages are installed only in this venv; pip constraints prevent replacing the working GPU packages. Do **not** install upstream's CUDA-only requirements over ROCm.

The integration pins GVHMR revision `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`. A small, recorded `backends/gvhmr/headless.patch` makes an unused mesh-KNN import lazy; it does not replace any inference operators with dummy functions. Only inference configs are registered. PyTorch3D's real pure-PyTorch transforms are installed, but its compiled mesh operations/renderer and DPVO are intentionally not built. Blender is the visualization target.

## 2. Obtain model files under your own licenses

**Licensed body model:** register/download from [SMPL-X](https://smpl-x.is.tue.mpg.de/). You need the **neutral SMPL-X `.npz`**, named `SMPLX_NEUTRAL.npz`, not a SMPL `.pkl` or SMPL-H file. Extract it yourself; no account password/token is needed by Framecraft.

**Trained model weights:** use the pretrained-model Google Drive folder linked by the [official GVHMR installation guide](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md):

https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD

Read the corresponding licenses before downloading. The HMR2 file must be **HMR2.0a**, as supplied by GVHMR. Do not substitute similarly named HMR2.0b or smaller ViTPose models. These are several large checkpoints, not a browser model download.

Place the extracted files here, relative to the project:

```text
.third-party/GVHMR/inputs/checkpoints/
├── body_models/smplx/SMPLX_NEUTRAL.npz
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

The headless path does **not** need DPVO or the separate SMPL rendering asset. We do not redistribute the licensed body model or download/accept these model licenses automatically.

Check what remains:

```powershell
.venv-gvhmr\Scripts\python.exe tools/gvhmr.py doctor --probe
```

`gpuAndImportsPassed` tests basic operators/imports only. `readyForAttempt` requires those tests and all five files. File existence/size is **not** checkpoint authenticity or successful inference. `inferenceVerified: false` deliberately prevents the doctor from implying an end-to-end test occurred.

## 3. Run a new capture

After reviewing the terms and placing trusted official files:

```powershell
.venv-gvhmr\Scripts\python.exe tools/gvhmr.py run 'input.mp4' 'new-dance.gvhmr.json' --static-camera --accept-noncommercial --trust-checkpoints
```

- `--static-camera` is appropriate only for a camera that actually stays still. Omit it for camera movement; SimpleVO estimates camera rotation. Scene cuts, zooming and tracking failures remain limitations.
- Optionally use `--focal-mm 24` when the **35mm-equivalent focal length** is known. This is not horizontal FOV in degrees. Browser pitch/FOV controls are not reused to overwrite GVHMR's trajectory.
- `--accept-noncommercial` acknowledges that your use and asset permissions meet the applicable licenses. It does not grant a commercial license.
- `--trust-checkpoints` permits upstream pickle-based checkpoint loading, which can execute code. Only use official, trusted weights. The compatibility override is limited to this CLI process/run. Never pass untrusted `.pt`, `.pth`, or `.ckpt` files.
- Original videos, captures and `.blend` scenes are never overwritten. Choose a new output ending in `.gvhmr.json`; `.mocap.json` identifies the other engine.
- Videos are timestamp-resampled to **30 fps**, as required by GVHMR, rather than simply relabeled at a different speed. Use one performer and clips up to **60 seconds / 1,800 output frames**. This is a limit, not a guarantee every such clip fits in memory.
- Each attempt uses a fresh `.gvhmr-runs/<id>` folder. The normalized local video and, on success, raw `hmr4d_results.pt` remain there for inspection. There is no stale cross-video cache reuse. No video is uploaded.
- Missing assets/dependencies stop the job with diagnostics; it never silently falls back to MediaPipe or labels an old result “GVHMR.”

## 4. Import into Blender

Run the updated `blender_importer.py` from Blender's Text Editor, then **File → Import → Framecraft Capture**, selecting the new `.gvhmr.json`.

The importer creates a separate **Framecraft GVHMR Rig** and **Framecraft GVHMR Root**. World pelvis translation is keyed once on the parent Empty; all 22 native local body rotations (including twist) and per-frame model joint offsets are retained. The first horizontal pelvis position is moved to the scene origin without changing vertical height or flattening jumps. GVHMR's Y-up coordinates are properly rotated to Blender Z-up; no camera-pitch heuristic is applied.

Disable **Import estimated ground root motion** to hold the first world pelvis position. The MediaPipe fitting/cleanup options do not alter native GVHMR rotations. Character retargeting remains separate; copying only rotations loses travel on the parent Empty.

This exports a body armature, not a skinned SMPL-X mesh or facial/finger animation. It is still monocular estimation, not measured ground-truth mocap.

## Regression checks

```powershell
python -m unittest discover -s tests -p 'test_*.py'
.venv-gvhmr\Scripts\python.exe tests/gvhmr_environment.py
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/blender_gvhmr.py
```

These exercise schema, timing, coordinate conversion, safety checks, and synthetic Blender kinematics—not pretrained model accuracy. The backend orchestration retains GVHMR's attribution and non-commercial notice in `backends/gvhmr/LICENSE`.
