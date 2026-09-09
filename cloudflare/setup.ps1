<#
.SYNOPSIS
    One-time setup of a Cloudflare NAMED tunnel (stable hostnames) for this project.

.DESCRIPTION
    Creates a named tunnel, points two DNS records at it, and stores everything
    the project needs inside the project folder so it can be copied to another
    machine and started with run.cmd - no re-setup, no changing URLs.

        studio.<domain>  ->  127.0.0.1:8100   (Journey 1 - Agent Studio)
        portal.<domain>  ->  127.0.0.1:8200   (Journey 2 - Device Runtime / phones)

    Unlike trycloudflare quick tunnels, these hostnames NEVER change, so the
    Android app's portal URL is set once. Unlike ngrok's free tier, there is no
    monthly transfer cap - multi-GB GGUF model downloads are fine.

.PARAMETER Domain
    A domain already added to your Cloudflare account, e.g. example.com

.EXAMPLE
    .\setup.ps1 -Domain example.com
    .\setup.ps1 -Domain example.com -StudioSub agents -PortalSub devices
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string] $Domain,
    [string] $TunnelName = "slm-agent-runtime",
    [string] $StudioSub  = "studio",
    [string] $PortalSub  = "portal",
    [switch] $Force
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_lib.ps1"

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }

$cf = Resolve-Cloudflared
Write-Host "cloudflared: $cf"

# --------------------------------------------------------------- 1. account login
Step 1 "Cloudflare account login"

$certPath = Join-Path $env:USERPROFILE ".cloudflared\cert.pem"
if ((Test-Path -LiteralPath $certPath) -and (-not $Force)) {
    Write-Host "    already logged in ($certPath)" -ForegroundColor Green
} else {
    Write-Host @"
    A browser window will open. Sign in to Cloudflare and pick the zone:
        $Domain
    This writes an account certificate to $certPath
"@
    & $cf tunnel login
    if ($LASTEXITCODE -ne 0) { throw "cloudflared tunnel login failed (exit $LASTEXITCODE)" }
    if (-not (Test-Path -LiteralPath $certPath)) { throw "Login did not produce $certPath" }
    Write-Host "    logged in" -ForegroundColor Green
}

# -------------------------------------------------------------- 2. create tunnel
Step 2 "Named tunnel '$TunnelName'"

$tunnelId = $null
$listJson = & $cf tunnel list --output json
if ($LASTEXITCODE -eq 0 -and $listJson) {
    $existing = ($listJson | ConvertFrom-Json) | Where-Object { $_.name -eq $TunnelName }
    if ($existing) {
        $tunnelId = $existing[0].id
        Write-Host "    reusing existing tunnel  id=$tunnelId" -ForegroundColor Green
    }
}

if (-not $tunnelId) {
    & $cf tunnel create $TunnelName
    if ($LASTEXITCODE -ne 0) { throw "cloudflared tunnel create failed (exit $LASTEXITCODE)" }
    $listJson = & $cf tunnel list --output json
    $created  = ($listJson | ConvertFrom-Json) | Where-Object { $_.name -eq $TunnelName }
    if (-not $created) { throw "Tunnel '$TunnelName' was created but is not listed." }
    $tunnelId = $created[0].id
    Write-Host "    created  id=$tunnelId" -ForegroundColor Green
}

# ------------------------------------------------- 3. credentials into the project
Step 3 "Copying tunnel credentials into the project folder"

$srcCreds = Join-Path $env:USERPROFILE ".cloudflared\$tunnelId.json"
$dstCreds = Join-Path $PSScriptRoot "$tunnelId.json"
if (Test-Path -LiteralPath $srcCreds) {
    Copy-Item -LiteralPath $srcCreds -Destination $dstCreds -Force
    Write-Host "    $dstCreds" -ForegroundColor Green
} elseif (Test-Path -LiteralPath $dstCreds) {
    Write-Host "    already present: $dstCreds" -ForegroundColor Green
} else {
    throw "Credentials file not found at $srcCreds - cannot continue."
}
Write-Host "    NOTE: this file is a SECRET (it authorises the tunnel). Never commit it." -ForegroundColor Yellow

# ------------------------------------------------------------ 4. record the config
Step 4 "Recording tunnel.json"

$studioHost = "$StudioSub.$Domain"
$portalHost = "$PortalSub.$Domain"

Write-TunnelConfig -Config ([pscustomobject]@{
    tunnel_name      = $TunnelName
    tunnel_id        = $tunnelId
    domain           = $Domain
    studio_hostname  = $studioHost
    portal_hostname  = $portalHost
    studio_url       = "https://$studioHost"
    portal_url       = "https://$portalHost"
    created_at       = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
})
Write-Host "    studio: https://$studioHost  ->  127.0.0.1:8100"
Write-Host "    portal: https://$portalHost  ->  127.0.0.1:8200"

# --------------------------------------------------------------- 5. DNS records
Step 5 "Pointing DNS at the tunnel"

foreach ($h in @($studioHost, $portalHost)) {
    & $cf tunnel route dns --overwrite-dns $TunnelName $h
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    WARNING: could not route $h (exit $LASTEXITCODE)." -ForegroundColor Yellow
        Write-Host "    Add it manually in the Cloudflare dashboard as a PROXIED CNAME:" -ForegroundColor Yellow
        Write-Host "        $h  ->  $tunnelId.cfargotunnel.com" -ForegroundColor Yellow
    } else {
        Write-Host "    routed $h" -ForegroundColor Green
    }
}

# ----------------------------------------------------------- 6. render + validate
Step 6 "Rendering config.yml and validating ingress"

$cfgYml = Render-TunnelConfigYml -Config (Get-TunnelConfig)
Write-Host "    $cfgYml"
& $cf tunnel --config $cfgYml ingress validate
if ($LASTEXITCODE -ne 0) { throw "Ingress validation failed - check $cfgYml" }

Write-Host @"

================================================================
 Setup complete.
================================================================
 Start the tunnel now:        .\run.cmd
 Start it on every boot:      .\install_task.ps1     (as Administrator)

 Stable URLs (these never change again):
   Studio  https://$studioHost
   Portal  https://$portalHost   <-- set this in the Android app (settings, Portal URL)

 Moving to another machine: copy the whole project folder (including
 cloudflare\$tunnelId.json) and run run.cmd. No re-setup needed.
================================================================
"@ -ForegroundColor Green
