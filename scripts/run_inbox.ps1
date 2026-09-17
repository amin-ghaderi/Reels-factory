$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Project venv not found. Run .\scripts\setup_windows.ps1 first."
}

& $venvPython (Join-Path $Root "scripts\healthcheck.py")
if ($LASTEXITCODE -ne 0) {
  throw "Healthcheck failed. Fix the issues above before processing inbox videos."
}

& $venvPython (Join-Path $Root "run_pipeline.py") inbox @args
exit $LASTEXITCODE
