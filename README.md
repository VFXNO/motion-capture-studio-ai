# Framecraft — AI Motion Capture Studio

A local-first prototype for extracting **one performer's body, hands, and facial motion** from recorded video, with a Blender importer. The browser MediaPipe workflow requires no account, API key, or paid service.

**Default RTMW3D-X engine (no account/SMPL assets):** choose a video and click **Extract motion**. The studio automatically prepares the isolated environment, downloads verified ONNX models, runs local Windows/AMD inference, and creates a downloadable Blender scene. No setup commands, manual model files, Git, Node.js, or importer script are needed for this workflow. Windows/Python 3.12 and installed Blender (for .blend creation) are required. Body/fingers and native face68; ground travel remains estimated. MediaPipe is an explicit alternative in the engine selector. See [RTMW3D.md](RTMW3D.md). Permissively licensed (Apache-2.0 model cards), not license-free.

**Optional GVHMR body/root backend:** a separate CLI retains native 22-joint body rotations and learned world travel. It requires licensed SMPL-X assets and pretrained checkpoints; its license is research/non-commercial. Pretrained inference is blocked by missing assets and has not been verified. It is not used by the studio's automatic RTMW3D workflow. See [GVHMR Setup.md](GVHMR%20Setup.md).

## Start on Windows

1. Double-click **Launch Studio.bat** in this folder. RTMW3D requires Python 3.12; its dependencies/models install automatically.
2. Open **http://127.0.0.1:8765** in current **Chrome or Edge**. If the browser opens before the server is ready, refresh it.
3. Choose an MP4/H.264 or WebM video. MOV works only if its codec is supported by the browser.
4. Leave **RTMW3D-X · local AI** selected, choose layers/frame rate, and click **Extract motion**. Setup, inference and Blender scene creation run automatically. Approximate height/FOV/tilt have defaults; adjust if known.
5. Scrub the timeline to review; click **Download Blender scene** and open it. JSON is also available. MediaPipe is an explicit alternative, with manual JSON import as below.

Or run `python serve.py` from this folder. Close the terminal or press Ctrl+C to stop the server. Do not open `index.html` with `file://`; ES modules and tracking workers need localhost.

## Bring a capture into Blender 4.2+

**Automatic RTMW3D:** download and open the generated `.blend`—skip the steps below. Blender is detected under its normal install folder or PATH, runs in a separate background process, and never opens your existing scenes.

**Optional manual JSON import / MediaPipe:**

1. Download **Blender importer**, or use `blender_importer.py` from this folder.
2. In Blender's **Scripting** workspace, open that Python file in the Text Editor, then click **Run Script** (Alt+P with the editor focused).
3. Choose **File → Import → Framecraft Capture (.mocap.json)**.
4. Select your downloaded capture. Optional: increase **Import every Nth sample** to reduce facial animation size.
5. Select **Framecraft Rig** and play the timeline. The importer adds a new collection without deleting existing objects; it changes the scene frame rate to match the capture.

The import includes:
- A body armature with shoulders, arms, legs, spine, and head; up to 30 finger bones.
- **Fit limbs, torso rotation, and support feet** is enabled by default: robust clip-wide limb lengths, analytic two-bone fitting, measured pelvis/chest yaw, and support-foot anchors. Disable it in the import dialog to compare the older aim-only behavior. This adds per-bone location as well as rotation curves; retargeting only rotations does not reproduce the full fitted capture.
- For grounded captures, an animated **Framecraft Root** parent controller that moves the entire body and face through the scene, plus a non-rendering floor outline at Z = 0. Root travel is kept at full sample rate even if facial/body import is decimated. The root is an **Empty object**, not a deform bone; preserve its transform or bake it into your character's root bone when retargeting/exporting.
- An animated facial **wire mesh**, not a finished face surface or a character facial rig.
- MediaPipe expression scores stored as animated `fc_*` custom properties on **Framecraft Face**. RTMW3D's native face68 has no expression scores.

**This is not automatic character retargeting.** Bind or retarget the captured armature to your character in Blender. Once retargeted, Blender can export the animation as FBX or glTF. This app itself exports a versioned `.mocap.json` file, not FBX/BVH.

## Ground-based root motion

With **Full body** and **Ground-based root motion** enabled, the app fits a perspective translation to the 2D/3D pose landmarks, estimates a flat floor from visible heels/toes, and exports a separate root trajectory. It estimates sideways travel, distance toward/away from the camera, and hip height above the floor rather than leaving the performer in place. The top-view **Ground path** panel shows the exported trajectory and foot-contact estimates as you scrub.

- **Horizontal lens FOV:** approximate horizontal camera field of view (default 60°). Incorrect settings especially affect depth/distance. This is not focal length in millimeters.
- **Camera downward tilt:** positive if the camera looks down toward the floor; 0° for a level camera (default 10°). The same rotation is applied to body, hand, and face coordinates during grounded Blender import.
- **Stabilize planted feet:** infers slow, low foot contacts; compensates root travel using retained contact anchors and corrects floor penetration. Airborne samples retain estimated jump height. This reduces sliding but is not per-leg IK or guaranteed foot locking.
- Blender's optional **Fit limbs, torso rotation, and support feet** additionally uses analytic per-leg fitting and world-space anchors during reported contacts. Reach limits and occluded or misclassified contacts can still cause slip; this is not a physical simulation or guaranteed floor reconstruction.
- **Fixed camera / flat floor only.** Panning, zooming, camera shake, roll, slopes, stairs, or hidden feet can invalidate the root. Camera motion is not removed and can appear as performer travel. No fully calibrated scene reconstruction is performed.
- Scale comes from RTMW3D's actor-height/FOV backprojection or MediaPipe's estimated body size, depending on the selected engine. Distances are approximate meters. Root XY starts near the first ground reference; Z is absolute estimated hip height above Z = 0, **not** a delta to add to the original hip height.
- Missing/invalid pose estimates hold the last valid root and clear foot locks. If there are no reliable feet or usable projections, the capture explicitly reports that root motion is unavailable and exports the remaining layers in place.
- Disable root motion for a traditional in-place capture, or uncheck **Import estimated ground root motion** in Blender. Old version-1 captures without root data import in place with a warning. Use **Repair older capture JSON** in the app to rebuild travel from their saved landmarks, without rerunning AI inference. It downloads a new `.grounded.mocap.json`; import that file instead of the original. The camera FOV and tilt settings apply to the repair too.

The additive version-1 export fields are `rootMotion` (method, settings, ground reference, coverage) and `frames[].root` (ground-space `position`, `confidence`, `contacts`, `status`). The root pass does not change body/hand landmarks; optional motion cleanup before it does. The Blender importer recenters body landmarks around the hips, rotates them to the calibrated ground axes, and applies the root **once** to both rig and face. JSON repair always creates a separate output, preserving the original file.

## Motion cleanup and accuracy limits

- The **heavy pose model** offers a higher-quality body estimator than the earlier full/balanced model, at higher download and processing cost. It does not upgrade the hand or face networks.
- Video samples retain up to 1,920 pixels on their longest edge. The neural models still resize internally; this cannot reveal missing fingers or reconstruct occluded limbs.
- **Motion cleanup** runs after tracking, using bidirectional, speed-adaptive filtering. It reduces high-frequency jitter without the old fixed one-way filter's consistent lag. It conservatively replaces isolated large landmark spikes, keeps missing tracks missing, and does not smooth across low-confidence gaps. Sharp real single-frame gestures can resemble outliers; set cleanup to 0% to preserve raw estimates.
- Root translation is fit with robust reprojection weighting so an inconsistent joint has less influence on the entire root trajectory. Lens settings, actor scale, camera movement, and model depth errors still matter.
- The Blender fitter estimates median limb lengths from confident samples, solves elbows/knees in tracked bend planes, and keys joint-head positions rather than relying only on changing bone directions. It clamps unreachable targets instead of stretching bones. This can intentionally move a joint away from an inconsistent raw detection.
- These are improvements in estimation, temporal consistency, and rig fitting—not a measured percentage improvement in real-world motion-capture accuracy. Ground-truth mocap or calibrated multi-camera footage would be needed for that claim.
- **Also clean saved motion** applies the selected cleanup strength during JSON repair. Older files may already have smoothing baked in; use a lower strength (e.g. 25%) to avoid over-filtering. New inference is required to benefit from the heavy model; repairing existing JSON does not rerun it.

Version-1 files can include `metadata.refinement` with cleanup settings and the number of isolated point outliers corrected. This count is not an accuracy score.

## Privacy and downloads

RTMW3D sends your chosen video only to the authenticated loopback API on this PC. The service runs isolated Python/DirectML inference and optional background Blender, with private inputs/results in `.rtmw3d-runs/<random-id>`. It binds only to 127.0.0.1, checks origin/host/session tokens, limits request sizes and runs one local job at a time. Private video/model/environment files are not static downloads. Cancel stops the job's process tree. Local files remain for inspection; originals are never overwritten.

Setup downloads pinned support code and model files from GitHub/Hugging Face and packages from PyPI; video is never sent to those hosts. MediaPipe instead processes browser-decoded frames in a worker and downloads JavaScript/WASM from jsDelivr and weights from Google Storage. Fonts load from Google Fonts. These normal download requests expose connection information, not footage. First use needs internet; cached RTMW3D model files are reused.

## Recording tips and limits

- One clearly visible person; fixed camera; good lighting; avoid motion blur and hidden limbs. The models are independent, so multi-person recordings may associate the wrong face or hands.
- Best results: full body on screen, hands large enough to see fingers, face facing the camera. A wide full-body shot often cannot resolve fine fingers or expressions.
- RTMW3D: maximum **60 seconds** at any rate and a 512 MB source file. MediaPipe: maximum **1,800 samples** (60 seconds at 30 fps, 75 seconds at 24 fps, or 150 seconds at 12 fps). Trim separate copies of longer clips.
- Input frames are reduced to a maximum dimension of 1,920 pixels for inference. Upscaling does not add detail.
- Selected frame rate is a **sampling rate**, not measured source FPS. Choosing above the source FPS can repeat decoded frames.
- Body world points are relative to the hips. Optional ground reconstruction adds an estimated root track for fixed-camera footage; it does not recover camera movement. With root disabled, the first detected feet establish approximate ground height for an in-place rig.
- Monocular depth, scale, occlusion, and bone twist are estimates. The enhanced importer uses median limb lengths and analytic limb fitting; the legacy mode uses the first detection and aim-only rotations. No physics or calibrated full-body twist recovery is included. A stabilized source landmark may not exactly match a fixed-length rig's sole.
- Hands are aligned at body wrists when body tracking is available. Face depth/scale is approximate and attached to the tracked nose. If body tracking is disabled, face/hands use fixed independent anchors.
- Missing detections are exported explicitly. The importer holds body/finger poses through gaps and hides the facial wire mesh during missing face samples. Handedness may be ambiguous when hands cross.
- Cleanup can attenuate fine movement or introduce endpoint bias. Face expressions are scores, not artist-authored blendshape geometry.
- Importing dense facial animation may take time and increase `.blend` file size. Use the import stride or disable facial import for lighter scenes.

## Development / tests

Normal studio use needs no Node/Git installation; the local AI dependencies install automatically. Only the optional CLI exporter and development/browser tests use Node. Dependency-specific Python tests run fully in `.venv-rtmw3d` and skip when packages are absent.

```powershell
node --test tests/*.test.mjs
python -m unittest discover -s tests -p "test_*.py"
```

Browser smoke tests (development only):

```powershell
npm.cmd install
npx.cmd playwright install chromium
npx.cmd playwright test
```

For a network-dependent integration test that loads all three real models and detects landmarks on Google's public example images, set `$env:REAL_MODELS='1'` before running Playwright. Browser UI tests otherwise use a mock tracker and a one-second generated test video; schema tests do not execute Blender itself.

`tracker.worker.mjs` runs MediaPipe; `core.mjs` defines sampling and the capture schema; `refinement.mjs` performs offline cleanup; `root-motion.mjs` reconstructs ground-based root travel; `app.mjs` handles video decoding and UI; `blender_importer.py` is a standalone Blender script. MediaPipe code/models are externally hosted and governed by their own licenses. Review their terms before redistributing them.

### Blender playback checks

Unlike schema tests, this runs the actual importer in a separate, factory-startup Blender process and checks evaluated root and rig world positions against every saved root sample:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python-exit-code 1 --python tests/blender_playback.py -- 'path\to\capture.grounded.mocap.json'
```

The CLI equivalent of the browser repair (requires Node) never overwrites its input or an existing output file:

```powershell
node tools/rebuild-root.mjs 'old.mocap.json' 'new.grounded.mocap.json' 60 10
```

Add a final cleanup strength (0–0.85) to refine saved landmarks too:

```powershell
node tools/rebuild-root.mjs 'old.mocap.json' 'new.refined.mocap.json' 60 10 0.25
```

`tests/blender_kinematics.py` checks fixed limb lengths, reach limits, torso yaw, support-foot anchors, and floor height inside Blender. `tests/blender_accuracy.py` reports rig-to-landmark fit and foot sliding/penetration on a capture; these are internal consistency measurements, not ground-truth accuracy.

With Playwright installed and the local app running, the development CLI can run a new high-quality video capture without overwriting an existing output:

```powershell
node tools/capture-video.mjs 'input.mp4' 'new-high-quality.mocap.json' 24
```
