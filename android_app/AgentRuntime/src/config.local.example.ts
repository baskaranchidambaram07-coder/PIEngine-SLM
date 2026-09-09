// ---------------------------------------------------------------------------
// LOCAL DEPLOYMENT SETTINGS — copy this file to `config.local.ts` and fill in
// your own addresses. `config.local.ts` is gitignored, so real hostnames of a
// running deployment never land in the public repo.
//
//     cp src/config.local.example.ts src/config.local.ts
//
// The app will not bundle without `config.local.ts` present.
// ---------------------------------------------------------------------------

/** Portal address baked into a fresh install (the Device Runtime, port 8200).
 *  Users can override it in-app under ⚙ Portal URL. With the Cloudflare named
 *  tunnel this is permanent: https://portal.<your-domain> */
export const DEFAULT_PORTAL = 'https://portal.your-domain.example';

/** Stable hosts that always know where the portal currently is. The app asks
 *  these for GET /api/portal-url whenever its saved portal URL stops
 *  answering, so a rotated tunnel URL cannot strand an installed handset.
 *  Order matters — first host that answers wins. Put the named-tunnel Studio
 *  host first: https://studio.<your-domain> */
export const DISCOVERY_URLS = [
  'https://studio.your-domain.example',
];
