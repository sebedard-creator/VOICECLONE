param(
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

Write-Host "VOICECLONE-QC setup Windows"
Write-Host "Recherche de Python $PythonVersion..."

$python = $null
try {
    $candidate = py "-$PythonVersion" -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -eq 0 -and $candidate) {
        $python = $candidate.Trim()
    }
} catch {
    $python = $null
}

if (-not $python) {
    throw "Python $PythonVersion est introuvable. Installe Python 3.10 ou 3.11 avant de continuer."
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg est introuvable dans le PATH. Installe ffmpeg avant le nettoyage audio."
}

Write-Host "Python utilise: $python"
& $python -m venv .venv

$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "..\requirements.txt")

Write-Host "Installation terminee."
Write-Host "Lance ensuite: .\scripts\run_app.ps1"

