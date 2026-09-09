# Shared helpers for the project's Cloudflare named-tunnel scripts.
# Dot-source this: . "$PSScriptRoot\_lib.ps1"
# Windows PowerShell 5.1 compatible (no ternary / null-coalescing operators).

Set-StrictMode -Version 2.0

function Get-ProjectRoot {
    # cloudflare/ lives directly under the project root.
    return (Split-Path -Parent $PSScriptRoot)
}

function Resolve-Cloudflared {
    <#  Finds the cloudflared binary, preferring the copy vendored in the
        project so a machine with nothing installed still works.  #>
    $root = Get-ProjectRoot
    $candidates = @(
        (Join-Path $root "tools\cloudflared.exe"),
        (Join-Path $PSScriptRoot "cloudflared.exe"),
        "C:\Program Files (x86)\cloudflared\cloudflared.exe",
        "C:\Program Files\cloudflared\cloudflared.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    $onPath = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    throw @"
cloudflared.exe not found.

Install it, or drop the binary at:  $root\tools\cloudflared.exe

  winget install --id Cloudflare.cloudflared
  # or download: https://github.com/cloudflare/cloudflared/releases/latest
"@
}

function Get-TunnelConfigPath { return (Join-Path $PSScriptRoot "tunnel.json") }

function Get-TunnelConfig {
    <#  Reads tunnel.json - the portable, machine-independent record of which
        named tunnel and hostnames this project uses.  #>
    $p = Get-TunnelConfigPath
    if (-not (Test-Path -LiteralPath $p)) {
        throw "Not set up yet: $p is missing. Run:  .\setup.ps1 -Domain <your-domain.com>"
    }
    return (Get-Content -LiteralPath $p -Raw | ConvertFrom-Json)
}

function Write-TunnelConfig {
    param([Parameter(Mandatory=$true)] $Config)
    $Config | ConvertTo-Json -Depth 6 |
        Out-File -LiteralPath (Get-TunnelConfigPath) -Encoding utf8
}

function Resolve-CredentialsFile {
    <#  The tunnel credentials JSON (contains the tunnel secret). Kept inside
        the project folder so the whole thing is copyable to another machine.  #>
    param([Parameter(Mandatory=$true)] $Config)
    return (Join-Path $PSScriptRoot ("{0}.json" -f $Config.tunnel_id))
}

function Render-TunnelConfigYml {
    <#  Renders config.template.yml -> config.yml with this machine's absolute
        paths. Done on every start so a copied project folder self-heals.  #>
    param([Parameter(Mandatory=$true)] $Config)

    $template = Join-Path $PSScriptRoot "config.template.yml"
    $outFile  = Join-Path $PSScriptRoot "config.yml"
    $creds    = Resolve-CredentialsFile -Config $Config
    $logDir   = Join-Path (Get-ProjectRoot) "logs"

    if (-not (Test-Path -LiteralPath $template)) { throw "Missing template: $template" }
    if (-not (Test-Path -LiteralPath $creds)) {
        throw @"
Tunnel credentials not found: $creds

On THIS machine the tunnel has never been created, and the credentials file was
not copied across. Either copy cloudflare\$($Config.tunnel_id).json from the
machine where you ran setup, or re-run:  .\setup.ps1 -Domain $($Config.domain)
"@
    }
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    $yml = Get-Content -LiteralPath $template -Raw
    $yml = $yml.Replace("{{TUNNEL_ID}}",        $Config.tunnel_id)
    $yml = $yml.Replace("{{CREDENTIALS_FILE}}", $creds)
    $yml = $yml.Replace("{{STUDIO_HOSTNAME}}",  $Config.studio_hostname)
    $yml = $yml.Replace("{{PORTAL_HOSTNAME}}",  $Config.portal_hostname)
    $yml = $yml.Replace("{{LOG_FILE}}",         (Join-Path $logDir "cloudflare-tunnel.log"))
    $yml | Out-File -LiteralPath $outFile -Encoding utf8

    return $outFile
}
