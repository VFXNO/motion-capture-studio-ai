# RTMW3D-X local backend

- RTMW3D / MMPose: Copyright OpenMMLab. Code: Apache License 2.0.
  <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d>
- rtmlib: Copyright its contributors. Apache License 2.0; source and full
  license are retained in `.third-party/rtmlib/`. Pinned revision:
  `03a1693e59e4f7cd84582c0fb30459b3bf18ad42`.
  <https://github.com/Tau-J/rtmlib>
- RTMW3D-X ONNX: converted/published by Soykaf; model card declares Apache-2.0.
  <https://huggingface.co/Soykaf/RTMW3D-x/tree/a4f0d54c52ee44ee4e1dda1936d714490c6234d2>
- YOLOX-M HumanArt ONNX: published by Tau-J; model card declares Apache-2.0.
  <https://huggingface.co/Tau-J/RTMPose/tree/cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd>
- RTMPose-X body7 384x288: OpenMMLab model, ONNX published by Tau-J in the
  same Apache-2.0 model repository. Checksum/revision pinned in `models.py`.
- MotionAGFormer: Soroush Mehraban, Vida Adeli, Babak Taati / TaatiTeam;
  upstream project is Apache-2.0. Full license: `MotionAGFormer-LICENSE.txt`.
  <https://github.com/TaatiTeam/MotionAGFormer/tree/4756fd1eb7cc73f0e991f091ff2280e030ab85f3>
  `temporal.py` adapts its COCO-to-H36M demo preprocessing and flip mapping,
  adding centered-square letterboxing, timestamp resampling and overlap blending.
- MotionAGFormer-B ONNX conversion by jaybanez:
  <https://huggingface.co/jaybanez/MotionAGFormer-3d-ONNX/tree/c1226b38f4f74e1ec966c169ec7de522a71ce34f>
  The conversion repository has no model card/license declaration of its own;
  upstream attribution/license is retained rather than inventing one.
  All **1,033 ONNX initializers** were verified exactly against the official
  Base H36M checkpoint (including transposed matrices). Official checkpoint:
  <https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view>
  SHA256 `eb15d088aadf193524a0d6c4a40680a324292db2781f3f45abe3400d5a4b343b`.
  Reference PyTorch vs ONNX on 243 actual dance frames: maximum absolute
  difference `0.000266`, mean `0.00000388` model units. Top-k graph adjacency
  is sensitive to floating-point ties. This verifies conversion, not pose accuracy.
  Runtime uses the checksum-pinned ONNX only; no pickle execution or Torch install.

Exact download URLs and published LFS SHA-256 digests: `models.py`, `temporal.py`.
These are permissively licensed, **not license-free or public-domain** assets.
Retain applicable Apache copyright, license and notice obligations when
redistributing them. Model-card declarations are not a guarantee that all
training data or third-party rights are cleared for every use.

No GVHMR, SMPL or SMPL-X code/assets/checkpoints are used by this backend.
