$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Reels Factory V1 setup"
Write-Host "Repository: $Root"

function Get-Python312 {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
      $exe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
      if ($LASTEXITCODE -eq 0 -and $exe) {
        return $exe.Trim()
      }
    } catch {
      # Fall through to other lookups.
    }
  }

  $candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:ProgramFiles\Python312\python.exe"
  )
  foreach ($path in $candidates) {
    if (Test-Path $path) {
      return $path
    }
  }

  if (Get-Command python -ErrorAction SilentlyContinue) {
    $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($ver -eq "3.12") {
      return (Get-Command python).Source
    }
  }
  return $null
}

function Test-Binary($name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

$python312 = Get-Python312
if (-not $python312) {
  throw "Python 3.12 was not found. Install it with: winget install -e --id Python.Python.3.12  then reopen PowerShell."
}
Write-Host "Python 3.12: $python312"

if (-not (Test-Binary "ffmpeg")) {
  Write-Host "FFmpeg was not found in PATH."
  Write-Host "Install it with: winget install -e --id Gyan.FFmpeg"
  throw "Install FFmpeg and reopen PowerShell before continuing."
}
if (-not (Test-Binary "ffprobe")) {
  Write-Host "ffprobe was not found in PATH (FFmpeg is incomplete)."
  Write-Host "Install a full FFmpeg build with: winget install -e --id Gyan.FFmpeg"
  throw "Install FFmpeg/ffprobe and reopen PowerShell before continuing."
}
Write-Host "FFmpeg: $((Get-Command ffmpeg).Source)"
Write-Host "FFprobe: $((Get-Command ffprobe).Source)"

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$needVenv = $true
if (Test-Path $venvPython) {
  $venvVer = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if ($venvVer -eq "3.12") {
    Write-Host "Using existing .venv (Python 3.12)"
    $needVenv = $false
  } else {
    Write-Host "Existing .venv is Python $venvVer; recreating with Python 3.12"
    Remove-Item -Recurse -Force (Join-Path $Root ".venv")
  }
}

if ($needVenv) {
  Write-Host "Creating .venv with Python 3.12"
  & $python312 -m venv (Join-Path $Root ".venv")
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $Root "requirements.txt")
$devReq = Join-Path $Root "requirements-dev.txt"
if (Test-Path $devReq) {
  & $venvPython -m pip install -r $devReq
}

& $venvPython (Join-Path $Root "run_pipeline.py") setup
& $venvPython (Join-Path $Root "scripts\healthcheck.py")
if ($LASTEXITCODE -ne 0) {
  throw "Healthcheck failed."
}

Write-Host ""
Write-Host "Done. Put a video in data\inbox, then run:"
Write-Host ".\scripts\run_inbox.ps1"
