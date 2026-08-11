$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Environnement virtuel introuvable. Lance d'abord .\scripts\setup_windows.ps1"
}

& $venvPython (Join-Path $PSScriptRoot "..\app.py")

