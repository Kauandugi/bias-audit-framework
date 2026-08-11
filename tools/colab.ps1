[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("doctor", "auth", "gpu-probe", "start", "mount-drive", "status", "smoke", "full", "stop")]
    [string]$Command = "doctor",

    [string]$Distro = "Ubuntu-24.04",
    [string]$Session = "biasaudit-t4",
    [string]$RepositoryRef = "tooling/colab-automation"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Notebook = Join-Path $RepoRoot "Framework_Auditoria_Viés_IA_Generativa.ipynb"
$Probe = Join-Path $RepoRoot "scripts\colab_gpu_probe.py"
$ArtifactRoot = Join-Path $RepoRoot "execucao\colab"

function Invoke-WslShell {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [switch]$AllowFailure
    )

    & wsl.exe -d $Distro -- bash -lc $Script
    $exitCode = $LASTEXITCODE
    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw "WSL command failed with exit code $exitCode."
    }
}

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $result = & wsl.exe -d $Distro -- wslpath -a $WindowsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not translate path to WSL: $WindowsPath"
    }
    return ($result | Select-Object -Last 1).Trim()
}

function Invoke-Colab {
    param(
        [Parameter(Mandatory = $true)][string]$Arguments,
        [switch]$AllowFailure
    )

    $script = 'export PATH="$HOME/.local/bin:$PATH"; colab --auth=oauth2 ' + $Arguments
    Invoke-WslShell -Script $script -AllowFailure:$AllowFailure
}

function Stop-ColabSession {
    Invoke-Colab -Arguments "stop -s $Session" -AllowFailure | Out-Null
}

function Invoke-NotebookRun {
    param([Parameter(Mandatory = $true)][bool]$RunFull)

    if (-not (Test-Path -LiteralPath $Notebook)) {
        throw "Official notebook not found: $Notebook"
    }

    $mode = if ($RunFull) { "full" } else { "smoke" }
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $runDirectory = Join-Path $ArtifactRoot "$mode-$timestamp"
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

    $runNotebook = Join-Path $runDirectory "BiasAuditFW-$mode.ipynb"
    $preamble = Join-Path $runDirectory "set_environment.py"
    Copy-Item -LiteralPath $Notebook -Destination $runNotebook -Force

    $runFullValue = if ($RunFull) { "true" } else { "false" }
    @"
import os
os.environ["BIASAUDIT_RUN_FULL"] = "$runFullValue"
os.environ["BIASAUDIT_REPOSITORY_REF"] = "$RepositoryRef"
print("BiasAuditFW environment configured:", {
    "run_full": os.environ["BIASAUDIT_RUN_FULL"],
    "repository_ref": os.environ["BIASAUDIT_REPOSITORY_REF"],
})
"@ | Set-Content -LiteralPath $preamble -Encoding utf8

    $preambleWsl = Convert-ToWslPath $preamble
    $notebookWsl = Convert-ToWslPath $runNotebook
    $logWsl = Convert-ToWslPath (Join-Path $runDirectory "session-log.ipynb")
    $jsonlWsl = Convert-ToWslPath (Join-Path $runDirectory "session-log.jsonl")

    try {
        Invoke-Colab -Arguments "exec -s $Session -f '$preambleWsl' --timeout 120"
        Invoke-Colab -Arguments "exec -s $Session -f '$notebookWsl' --timeout 14400"
        Invoke-Colab -Arguments "log -s $Session -o '$logWsl'"
        Invoke-Colab -Arguments "log -s $Session -o '$jsonlWsl'"
        Write-Host "Execution completed. Local logs: $runDirectory"
        Write-Host "Scientific outputs remain in the configured Google Drive RUN_OUTPUT directory."
    }
    finally {
        Stop-ColabSession
    }
}

switch ($Command) {
    "doctor" {
        Invoke-WslShell -Script 'set -e; printf "Python: "; python3 --version; printf "uv: "; uv --version; printf "Colab CLI: "; colab version'
    }
    "auth" {
        Invoke-Colab -Arguments "sessions"
    }
    "gpu-probe" {
        $probeWsl = Convert-ToWslPath $Probe
        Invoke-Colab -Arguments "run --gpu T4 -s biasaudit-probe --timeout 300 '$probeWsl'"
    }
    "start" {
        Invoke-Colab -Arguments "new -s $Session --gpu T4"
    }
    "mount-drive" {
        Invoke-Colab -Arguments "drivemount -s $Session"
    }
    "status" {
        Invoke-Colab -Arguments "sessions"
        Invoke-Colab -Arguments "status -s $Session" -AllowFailure | Out-Null
    }
    "smoke" {
        Invoke-NotebookRun -RunFull $false
    }
    "full" {
        Invoke-NotebookRun -RunFull $true
    }
    "stop" {
        Stop-ColabSession
    }
}
