$ErrorActionPreference = "Continue"

Write-Host "Python launcher:"
py -0p

Write-Host "`nffmpeg:"
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
    Write-Host $ffmpeg.Source
} else {
    Write-Warning "ffmpeg introuvable dans le PATH."
}

Write-Host "`nDisque Y:"
if (Test-Path Y:\) {
    Write-Host "Y:\ est disponible."
} else {
    Write-Warning "Y:\ est indisponible."
}

Write-Host "`nDossier runtime:"
if (Test-Path Y:\VOICECLONE) {
    Get-ChildItem Y:\VOICECLONE | Select-Object Name,Mode
} else {
    Write-Warning "Y:\VOICECLONE n'existe pas encore."
}

