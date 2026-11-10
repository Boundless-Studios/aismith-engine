param(
  [switch]$ForceTorch,
  [switch]$InstallBlender
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $npm) { throw 'Node.js 20+ (including npm) is required. Install the Node.js LTS release and rerun AISmith 3D.' }
$venv = Join-Path $root '.venv'
$pythonLauncher = (Get-Command py -ErrorAction SilentlyContinue)
$systemPython = if ($pythonLauncher) { & $pythonLauncher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null } else { (Get-Command python).Source }
if (-not $systemPython -or -not (Test-Path $systemPython)) { throw 'Python 3.11 or 3.12 is required. Install Python from python.org and rerun AISmith 3D.' }

if (-not (Test-Path (Join-Path $venv 'Scripts/python.exe'))) {
  Write-Host "Creating dedicated AISmith 3D environment at $venv" -ForegroundColor Cyan
  & $systemPython -m venv $venv
}
$python = Join-Path $venv 'Scripts/python.exe'
function Invoke-Pip {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PipArgs)
  & $python -m pip @PipArgs
  if ($LASTEXITCODE -ne 0) { throw "pip failed: $($PipArgs -join ' ')" }
}

Invoke-Pip -PipArgs @('install','--upgrade','--no-cache-dir','pip','setuptools','wheel')

$gpuName = ''
try { $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim() } catch {}
$hasNvidia = $LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($gpuName)
$torchReady = $false
try { & $python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null; $torchReady = $LASTEXITCODE -eq 0 } catch {}

if ($ForceTorch -or -not $torchReady) {
  if ($hasNvidia) {
    $torchIndex = if ($env:PYTORCH_INDEX_URL) { $env:PYTORCH_INDEX_URL } else { 'https://download.pytorch.org/whl/cu128' }
    Write-Host "Installing CUDA PyTorch for $gpuName from $torchIndex" -ForegroundColor Cyan
    Invoke-Pip -PipArgs @('install','--upgrade','--no-cache-dir','torch==2.8.0','torchvision==0.23.0','torchaudio==2.8.0','--index-url',$torchIndex)
  } else {
    Write-Host 'No NVIDIA GPU detected; installing the CPU-compatible Python stack. GPU workers will remain unavailable.' -ForegroundColor Yellow
    Invoke-Pip -PipArgs @('install','--upgrade','--no-cache-dir','torch==2.8.0','torchvision==0.23.0','torchaudio==2.8.0')
  }
}

Invoke-Pip -PipArgs @('install','--upgrade','--no-cache-dir','-e',$root)
Invoke-Pip -PipArgs @('install','--upgrade','--no-cache-dir','huggingface_hub','requests','scipy','open3d','plotly','tqdm')

# The bundled CUDA extensions are built against NumPy 1.26.  Pin SciPy to the
# last compatible release instead of allowing its newer NumPy-2-only wheel.
Invoke-Pip -PipArgs @('install','--force-reinstall','--no-cache-dir','numpy==1.26.4','scipy==1.14.1','--no-deps')

Write-Host 'Installing the web application dependencies...' -ForegroundColor Cyan
& $npm.Source --prefix (Join-Path $root 'frontend') ci
if ($LASTEXITCODE -ne 0) { throw 'npm ci failed for the frontend.' }

$blenderCandidates = @(
  'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe',
  'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe'
)
$blenderFound = ($blenderCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
if (-not $blenderFound -and (Get-Command blender -ErrorAction SilentlyContinue)) { $blenderFound = (Get-Command blender).Source }
if (-not $blenderFound -and $InstallBlender) {
  $winget = Get-Command winget -ErrorAction SilentlyContinue
  if ($winget) {
    Write-Host 'Blender 4.2+ was not found; installing it with winget for native GLB and texture-baking workers.' -ForegroundColor Cyan
    try { & $winget.Source install --id BlenderFoundation.Blender --exact --silent --accept-package-agreements --accept-source-agreements } catch { Write-Warning "Automatic Blender install failed: $($_.Exception.Message)" }
  } else {
    Write-Warning 'Blender 4.2+ was not found and winget is unavailable. Install Blender before using texture baking.'
  }
} elseif (-not $blenderFound) {
  Write-Warning 'Blender 4.2+ was not found. It is only required for Texture Paint and Blender-assisted mesh conversion. Install it yourself or rerun with -InstallBlender.'
}

Write-Host "AISmith 3D environment ready: $python" -ForegroundColor Green
