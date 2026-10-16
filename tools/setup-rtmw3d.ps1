param([string]$Python = "")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (!$Python) { $Python = (& py -3.12 -c "import sys; print(sys.executable)").Trim() }
& $Python (Join-Path $root "tools\rtmw3d.py") setup
if ($LASTEXITCODE -ne 0) { throw "Automatic RTMW3D setup failed." }
