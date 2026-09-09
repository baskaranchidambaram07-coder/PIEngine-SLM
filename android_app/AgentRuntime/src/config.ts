import ReactNativeBlobUtil from 'react-native-blob-util';

export const DOC_DIR = ReactNativeBlobUtil.fs.dirs.DocumentDir;
export const MODELS_DIR = `${DOC_DIR}/models`;
export const AGENTS_DIR = `${DOC_DIR}/agents`;

// Same embedder family as the Studio (bge-small-en-v1.5). The Studio embeds
// passages with fastembed ONNX; llama.cpp GGUF embeddings were verified
// cross-compatible (cosine 0.9998+ on identical texts).
export const EMBEDDER_FILE = 'bge-small-en-v1.5-q8_0.gguf';
export const EMBEDDER_URL =
  'https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf';
export const EMBEDDING_DIM = 384;
// bge query-side instruction — must match fastembed's query_embed behaviour
export const QUERY_PREFIX =
  'Represent this sentence for searching relevant passages: ';

export const MAX_UPLOAD_BYTES = 2 * 1024 * 1024; // 2 MB inline-KB limit
export const MAX_TOOL_ROUNDS = 3;

// Deployment-specific addresses live in config.local.ts, which is gitignored
// so a public repo never carries the hostnames of a running deployment.
// New checkout:  cp src/config.local.example.ts src/config.local.ts
//
//   DEFAULT_PORTAL  portal baked into a fresh install (overridable in ⚙)
//   DISCOVERY_URLS  stable hosts asked for GET /api/portal-url when the saved
//                   portal URL stops answering, so a rotated quick-tunnel URL
//                   cannot strand an installed handset. First to answer wins;
//                   put https://studio.<your-domain> first once the named
//                   tunnel is set up — it and the portal it reports are permanent.
export { DEFAULT_PORTAL, DISCOVERY_URLS } from './config.local';

// Probe/discovery budgets. Kept short: this runs before the agent store loads.
export const PORTAL_PROBE_MS = 6000;
export const DISCOVERY_TIMEOUT_MS = 8000;
// ngrok free tier shows an HTML interstitial unless the skip header is present.
// Accept-Encoding: identity prevents ReactNativeBlobUtil's false "Download
// interrupted" on gzip-compressed responses (it compares decompressed bytes
// written against the compressed Content-Length).
export const PORTAL_HEADERS = {
  'ngrok-skip-browser-warning': '1',
  'Accept-Encoding': 'identity',
};
