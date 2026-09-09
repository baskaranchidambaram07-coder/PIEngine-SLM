from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
BUNDLES_DIR = ROOT / "bundles"
LLAMA_DIR = ROOT / "llama"
EMBEDDER_CACHE = MODELS_DIR / "embedder_cache"
# Shared governance/telemetry store — the runtime (Journey 2) + the Android app
# write usage events here; the Studio (Journey 1) reads them for the Governance
# dashboard. One file, on the same machine, so both processes reach it.
TELEMETRY_DB = ROOT / "telemetry.db"

for _d in (MODELS_DIR, BUNDLES_DIR, EMBEDDER_CACHE):
    _d.mkdir(parents=True, exist_ok=True)
