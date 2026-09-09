<#
.SYNOPSIS
    Registers the named tunnel as a Windows scheduled task so it starts on boot
    and survives logout - matching how slm-studio / slm-runtime already run.
.DESCRIPTION
    Must be run from an ELEVATED PowerShell (it registers a SYSTEM task).
    Paths are derived from this script's location, so it works on any machine.
#>
[CmdletBinding()]
param([string] $TaskName = "slm-cf-tunnel")

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_lib.ps1"

$config = Get-TunnelConfig          # fails fast if setup.ps1 has not been run
$runCmd = Join-Path $PSScriptRoot "run.cmd"
if (-not (Test-Path -LiteralPath $runCmd)) { throw "Missing $runCmd" }

Write-Host "Registering scheduled task '$TaskName' -> $runCmd"
schtasks /Create /TN $TaskName /TR "`"$runCmd`"" /SC ONSTART /RU SYSTEM /RL HIGHEST /F
if ($LASTEXITCODE -ne 0) {
    throw "schtasks failed (exit $LASTEXITCODE). Run this script as Administrator."
}

schtasks /Run /TN $TaskName | Out-Null
Write-Host @"
Task '$TaskName' registered and started.

  Stop:     schtasks /End    /TN $TaskName
  Start:    schtasks /Run    /TN $TaskName
  Remove:   schtasks /Delete /TN $TaskName /F
  Log:      $(Join-Path (Get-ProjectRoot) 'logs\cloudflare-tunnel.log')

  Studio    $($config.studio_url)
  Portal    $($config.portal_url)
"@ -ForegroundColor Green
