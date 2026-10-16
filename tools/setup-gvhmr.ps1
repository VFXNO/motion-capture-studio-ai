# Reuses an existing working PyTorch installation read-only. All new packages
# are installed in a dedicated venv. Does not download model weights/assets.
param([string]$Python = "")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (!$Python) {
    $Python = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required, or supply -Python PATH." }
}
$versions = & $Python -c "import json, torch, torchvision; assert torch.cuda.is_available(), 'A working GPU PyTorch installation is required'; print(json.dumps({'torch':torch.__version__, 'torchvision':torchvision.__version__}))"
if ($LASTEXITCODE -ne 0) { throw "Install/test PyTorch for your GPU first. No packages have been changed." }
$versions = $versions | ConvertFrom-Json
$venv = Join-Path $root ".venv-gvhmr"
if (!(Test-Path $venv)) {
    & $Python -m venv --system-site-packages $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create environment." }
}
$exe = Join-Path $venv "Scripts\python.exe"
if (!(Test-Path $exe)) { throw "Existing venv is not a Windows Python environment: $venv" }
$constraints = Join-Path $venv "gpu-constraints.txt"
@("torch==$($versions.torch)", "torchvision==$($versions.torchvision)") | Set-Content -Encoding ASCII $constraints
& $exe -m pip install -c $constraints -r (Join-Path $root "backends\gvhmr\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Your base PyTorch was not modified." }
$oldNoExtension = $env:PYTORCH3D_NO_EXTENSION
try {
    # GVHMR inference uses pure-Torch transforms, not PyTorch3D's mesh renderer.
    # Compiled rasterization/ops are deliberately unavailable in this venv.
    $env:PYTORCH3D_NO_EXTENSION = "1"
    & $exe -m pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch3D transforms." }
} finally { $env:PYTORCH3D_NO_EXTENSION = $oldNoExtension }
$repo = Join-Path $root ".third-party\GVHMR"
if (!(Test-Path $repo)) {
    git clone https://github.com/zju3dv/GVHMR.git $repo
    if ($LASTEXITCODE -ne 0) { throw "Could not clone GVHMR." }
    git -C $repo checkout ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
    if ($LASTEXITCODE -ne 0) { throw "Could not select tested GVHMR revision." }
}
$patch = Join-Path $root "backends\gvhmr\headless.patch"
$oldPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"  # A failed reverse check means not yet patched.
    git -C $repo apply --reverse --check $patch 2>$null
    $alreadyPatched = $LASTEXITCODE -eq 0
} finally { $ErrorActionPreference = $oldPreference }
if (!$alreadyPatched) {
    git -C $repo apply --check $patch
    if ($LASTEXITCODE -ne 0) { throw "Headless patch conflicts with source changes. No source changes have been discarded." }
    git -C $repo apply $patch
    if ($LASTEXITCODE -ne 0) { throw "Could not apply headless import patch." }
}
foreach ($directory in @("body_models\smplx", "gvhmr", "hmr2", "vitpose", "yolo")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo "inputs\checkpoints\$directory") | Out-Null
}
& $exe (Join-Path $root "tools\gvhmr.py") doctor --probe
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setup is incomplete. See the diagnostics and GVHMR Setup.md; licensed model files are not downloaded automatically."
}
