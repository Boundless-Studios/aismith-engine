param(
  [switch]$Install,
  [switch]$InstallBlender,
  [switch]$RestartApi
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=][^=]*)=(.*)$') {
      $envName = $Matches[1].Trim()
      $envValue = $Matches[2].Trim().Trim('"')
      Set-Item -Path "Env:$envName" -Value $envValue
    }
  }
}

$venvPython = Join-Path $root '.venv/Scripts/python.exe'
$apiState = Join-Path $root 'studio-data/api-process.json'
 $environmentReady = $false
if (Test-Path $venvPython) {
  & $venvPython -c "import fastapi, uvicorn, torch" 2>$null
  $environmentReady = $LASTEXITCODE -eq 0
}
 $frontendReady = Test-Path (Join-Path $root 'frontend/node_modules')
if (-not $environmentReady -or -not $frontendReady -or $Install) {
  Write-Host 'AISmith 3D environment is missing or incomplete; running the local setup.' -ForegroundColor Cyan
  $setupArgs = @()
  if ($InstallBlender) { $setupArgs += '-InstallBlender' }
  & (Join-Path $root 'scripts/setup.ps1') @setupArgs
}
if (-not (Test-Path $venvPython)) { throw 'AISmith 3D dedicated environment was not created.' }

# Replacing a prior launcher-owned API is safe even when it no longer owns the
# port. This prevents invisible stale workers from accumulating between runs.
if (Test-Path $apiState) {
  try {
    $savedApi = Get-Content $apiState -Raw | ConvertFrom-Json
    $savedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($savedApi.pid)" -ErrorAction SilentlyContinue
    if ($savedProcess -and $savedProcess.ExecutablePath -eq $venvPython -and $savedProcess.CommandLine -match 'backend\.app\.main:app') {
      Stop-Process -Id $savedApi.pid -Force
    }
  } catch {}
  Remove-Item -LiteralPath $apiState -Force -ErrorAction SilentlyContinue
}

$apiListener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($apiListener) {
  $apiProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($apiListener.OwningProcess)"
  $isAISmithApi = $apiProcess.CommandLine -match 'backend\.app\.main:app'
  if ($isAISmithApi -and $RestartApi) {
    Stop-Process -Id $apiListener.OwningProcess -Force
  } elseif ($isAISmithApi) {
    throw "An AISmith 3D API is already listening on port 8000 using: $($apiProcess.ExecutablePath). Restart with .\start.ps1 -RestartApi so it uses the dedicated .venv."
  } else {
    throw "Port 8000 is already in use by $($apiProcess.Name). Stop that application or change its port before starting AISmith 3D."
  }
}

Write-Host 'Starting AISmith 3D API on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
$api = Start-Process -WindowStyle Hidden -PassThru -FilePath $venvPython -ArgumentList '-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $root
@{ pid = $api.Id } | ConvertTo-Json | Set-Content -LiteralPath $apiState -Encoding utf8
Write-Host 'Starting AISmith 3D UI on http://127.0.0.1:5173 ...' -ForegroundColor Cyan
npm.cmd --prefix frontend run dev -- --host 127.0.0.1
