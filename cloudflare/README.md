# Cloudflare named tunnel — stable public URLs

Gives this project **two permanent HTTPS hostnames** that survive reboots,
tunnel restarts and moving the project to a different machine:

| Hostname | Serves | Local port |
|---|---|---|
| `https://studio.<your-domain>` | Journey 1 — Agent Studio, governance dashboard, landing page | 8100 |
| `https://portal.<your-domain>` | Journey 2 — Device Runtime, the phone-facing portal (bundles, GGUF models, APK) | 8200 |

## Why a *named* tunnel

| | trycloudflare quick tunnel | ngrok free | **named tunnel** |
|---|---|---|---|
| URL stability | new random URL every restart | static domain | **permanent** |
| Transfer cap | none | ~1 GB/month | **none** |
| Multi-GB GGUF model downloads | ok | **fails mid-download** | **ok** |
| Auto-start on boot | yes | yes | yes |
| Needs a domain | no | no | **yes** |

The 1 GB ngrok cap is exactly why Journey 2 was moved off ngrok: a single
Qwen3-1.7B Q4_K_M download is ~1.03 GB, so one phone install could exhaust the
month. Cloudflare tunnels have no such cap.

## One-time setup

Prerequisites — both are on Cloudflare's free plan:

1. A **Cloudflare account** — https://dash.cloudflare.com/sign-up
2. A **domain added to that account** ("zone"). Either register one through
   Cloudflare (~$10/yr, at cost) or point an existing domain's nameservers at
   Cloudflare. The dashboard walks through it; DNS propagation is usually
   minutes. A stable hostname is impossible without this — it is the one part
   that cannot be faked.

Then, from this folder:

```powershell
.\setup.ps1 -Domain your-domain.com
```

That will:

1. open a browser for `cloudflared tunnel login` (pick your zone),
2. create a named tunnel `slm-agent-runtime`,
3. copy its credentials into this folder,
4. write `tunnel.json` (the portable record of tunnel id + hostnames),
5. create proxied CNAMEs for `studio.` and `portal.`,
6. render and validate `config.yml`.

Custom subdomains: `.\setup.ps1 -Domain your-domain.com -StudioSub agents -PortalSub devices`

## Running it

```powershell
.\run.cmd                  # foreground
.\install_task.ps1         # register as a boot-start SYSTEM task (run as Administrator)
```

`install_task.ps1` creates the scheduled task `slm-cf-tunnel`, matching how
`slm-studio` and `slm-runtime` already run.

```powershell
schtasks /Run    /TN slm-cf-tunnel      # start
schtasks /End    /TN slm-cf-tunnel      # stop
schtasks /Delete /TN slm-cf-tunnel /F   # remove
```

Log: `..\logs\cloudflare-tunnel.log`

## Moving to another machine

The whole point of keeping everything in the project folder:

1. Copy the project folder across (**including `cloudflare\<tunnel-id>.json`**).
2. Make sure `cloudflared.exe` exists — `..\tools\cloudflared.exe`, on `PATH`,
   or installed via `winget install --id Cloudflare.cloudflared`.
3. `.\run.cmd`

No second login, no new DNS records, and **the URLs do not change** — so no
handset needs reconfiguring. `run.ps1` re-renders `config.yml` with the new
machine's paths on every start, so a different install drive is fine.

Only one machine should run the tunnel at a time for this project. (Cloudflare
does load-balance multiple replicas of one tunnel, but both machines would need
the Studio and Runtime services up, and requests would land on either one.)

## Files

| File | Purpose | Copy to other machines? |
|---|---|---|
| `setup.ps1` | one-time account/tunnel/DNS setup | yes |
| `run.ps1`, `run.cmd` | start the tunnel | yes |
| `install_task.ps1` | boot-start scheduled task | yes |
| `_lib.ps1` | shared path/config helpers | yes |
| `config.template.yml` | ingress rules — **edit this one** | yes |
| `tunnel.json` | tunnel id + hostnames (generated) | yes |
| `<tunnel-id>.json` | **tunnel secret** (generated) | yes, but never commit it |
| `config.yml` | rendered per machine on each start (generated) | no |

## How the rest of the project uses this

- `studio/app.py` reads `tunnel.json` and serves `GET /api/portal-url`, so the
  landing page and the Android app always learn the current portal address.
  Named-tunnel hostnames take priority over any quick-tunnel URL.
- The Android app (`src/config.ts` → `DISCOVERY_URLS`) re-resolves its portal
  URL from that endpoint whenever the saved one stops answering. After setup,
  put `https://studio.<your-domain>` at the front of that list.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `tunnel.json is missing` | setup has not run on any machine yet — run `setup.ps1 -Domain ...` |
| `Tunnel credentials not found` | `cloudflare\<tunnel-id>.json` was not copied across; copy it or re-run setup |
| Hostname returns Cloudflare error 1033 | tunnel is not running — `schtasks /Run /TN slm-cf-tunnel` |
| Hostname returns 502 | tunnel is up but the local service is down — `schtasks /Run /TN slm-runtime` |
| DNS route step warned | add a **proxied** CNAME by hand: `portal` → `<tunnel-id>.cfargotunnel.com` |
| Model download stalls | check `..\logs\cloudflare-tunnel.log`; the tunnel retries automatically |
