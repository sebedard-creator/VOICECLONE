$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$settingsPath = Join-Path $projectRoot "config\settings.json"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"
$logsDir = Join-Path $projectRoot "logs"
$logPath = Join-Path $logsDir "voiceclone_server.log"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$mutex = New-Object System.Threading.Mutex($false, "Global\VOICECLONE_QC_SERVER_KEEPALIVE")
if (-not $mutex.WaitOne(0, $false)) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format s) keepalive deja actif; sortie."
    exit 0
}

function Write-Log {
    param([string]$Message)
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format s) $Message"
}

function Get-AppPort {
    try {
        $settings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return [int]$settings.gradio.port
    } catch {
        return 7860
    }
}

function Test-TcpPort {
    param(
        [string]$HostName = "127.0.0.1",
        [int]$Port
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(1000, $false)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

try {
    Write-Log "keepalive demarre."

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Python virtuel introuvable: $venvPython"
    }
    if (-not (Test-Path -LiteralPath $appPath)) {
        throw "Application introuvable: $appPath"
    }

    while ($true) {
        $port = Get-AppPort

        if (Test-TcpPort -Port $port) {
            Write-Log "port $port deja actif; verification dans 30 secondes."
            Start-Sleep -Seconds 30
            continue
        }

        Write-Log "demarrage VOICECLONE-QC sur port $port."
        try {
            Push-Location $projectRoot
            $cmdLine = "`"$venvPython`" `"$appPath`" 2>&1"
            & cmd.exe /d /c $cmdLine | ForEach-Object {
                Add-Content -LiteralPath $logPath -Value $_
            }
            $exitCode = $LASTEXITCODE
            Write-Log "serveur termine avec code $exitCode; redemarrage dans 10 secondes."
        } catch {
            Write-Log "erreur serveur: $($_.Exception.Message)"
        } finally {
            Pop-Location
        }

        Start-Sleep -Seconds 10
    }
} finally {
    $mutex.ReleaseMutex() | Out-Null
    $mutex.Dispose()
}
