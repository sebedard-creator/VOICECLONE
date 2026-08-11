[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$settingsPath = Join-Path $projectRoot "config\settings.json"
$keepalivePath = Join-Path $PSScriptRoot "keepalive_server.ps1"

function Get-AppPort {
    try {
        $settings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return [int]$settings.gradio.port
    } catch {
        return 7860
    }
}

function Stop-KeepAlive {
    $processes = Get-CimInstance Win32_Process |
        Where-Object {
            ($_.Name -in @("powershell.exe", "pwsh.exe")) -and
            ($_.CommandLine) -and
            ($_.CommandLine -like "*$keepalivePath*")
        }

    foreach ($process in $processes) {
        Write-Host "Arret keepalive PID $($process.ProcessId)"
        if ($PSCmdlet.ShouldProcess("keepalive PID $($process.ProcessId)", "Stop-Process")) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Stop-PortOwner {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    $ownerPids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($ownerPid in $ownerPids) {
        if (-not $ownerPid) {
            continue
        }

        $process = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            continue
        }

        Write-Host "Arret serveur sur port $Port PID $ownerPid ($($process.ProcessName))"
        if ($PSCmdlet.ShouldProcess("serveur PID $ownerPid sur port $Port", "Stop-Process")) {
            Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
        }
    }
}

$port = Get-AppPort
Write-Host "VOICECLONE-QC stop_server"
Write-Host "Projet: $projectRoot"
Write-Host "Port: $port"

Stop-KeepAlive
Start-Sleep -Milliseconds 500
Stop-PortOwner -Port $port

Write-Host "Termine."
