<#
.SYNOPSIS
    Starts the project's Cloudflare named tunnel (stable hostnames).
.DESCRIPTION
    Re-renders config.yml from the template using THIS machine's paths, then
    runs the tunnel in the foreground. Safe to run on a freshly copied project
    folder: everything needed lives in cloudflare\.
#>
[CmdletBinding()]
param([switch] $ValidateOnly)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_lib.ps1"

$cf     = Resolve-Cloudflared
$config = Get-TunnelConfig
$cfgYml = Render-TunnelConfigYml -Config $config

Write-Host "tunnel : $($config.tunnel_name)  ($($config.tunnel_id))"
Write-Host "studio : $($config.studio_url)  -> 127.0.0.1:8100"
Write-Host "portal : $($config.portal_url)  -> 127.0.0.1:8200"
Write-Host "config : $cfgYml"

& $cf tunnel --config $cfgYml ingress validate
if ($LASTEXITCODE -ne 0) { throw "Ingress validation failed - check $cfgYml" }
if ($ValidateOnly) { Write-Host "ingress OK" -ForegroundColor Green; return }

& $cf tunnel --config $cfgYml run $config.tunnel_name
exit $LASTEXITCODE
