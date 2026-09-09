"""Curated SLM catalog for the Studio model picker.

Every entry is a 4-bit GGUF that fits comfortably in the RAM budget of the
target devices (iPhone 14+, Galaxy S23+, i.e. >= 6-8GB with ~2-3GB usable
for an app). download_url lets the Studio (or the device runtime) pull the
file on demand; license notes flag gated repos.
"""

MODEL_CATALOG = [
    {
        "id": "qwen3-1.7b-q4_k_m",
        "name": "Qwen3 1.7B (Q4_K_M)",
        "file": "Qwen3-1.7B-Q4_K_M.gguf",
        "download_url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        "size_gb": 1.03,
        "size_bytes": 1107409472,
        "min_device_ram_gb": 4,
        "context_length": 8192,
        "license": "Apache-2.0 (ungated)",
        "notes": "Default. Strong tool-calling for its size; hybrid thinking mode (disabled via /no_think). Best quality/speed balance for iPhone 14 / S23 class devices.",
        "family": "qwen3",
    },
    {
        "id": "qwen3-0.6b-q4_k_m",
        "name": "Qwen3 0.6B (Q4_K_M)",
        "file": "Qwen3-0.6B-Q4_K_M.gguf",
        "download_url": "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
        "size_gb": 0.37,
        "size_bytes": 396705472,
        "min_device_ram_gb": 3,
        "context_length": 8192,
        "license": "Apache-2.0 (ungated)",
        "notes": "Fastest option — 30+ tok/s on recent phones. Good for classification / extraction agents; weaker at multi-step reasoning.",
        "family": "qwen3",
    },
    {
        "id": "qwen3-4b-instruct-2507-q4_k_m",
        "name": "Qwen3 4B Instruct 2507 (Q4_K_M)",
        "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "download_url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "size_gb": 2.33,
        "size_bytes": 2497281120,
        "min_device_ram_gb": 8,
        "context_length": 16384,
        "license": "Apache-2.0 (ungated)",
        "notes": "Highest quality that still runs on 8GB-RAM flagships (iPhone 15 Pro+, S23 Ultra+). Non-thinking instruct variant.",
        "family": "qwen3",
    },
    {
        "id": "llama-3.2-3b-instruct-q4_k_m",
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "download_url": "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 1.88,
        "size_bytes": 2019377600,
        "min_device_ram_gb": 6,
        "context_length": 8192,
        "license": "Llama 3.2 Community License",
        "notes": "Good general assistant; widely benchmarked on mobile (ExecuTorch reference model).",
        "family": "llama3",
    },
    {
        "id": "gemma-3-4b-it-q4_k_m",
        "name": "Gemma 3 4B IT (Q4_K_M)",
        "file": "gemma-3-4b-it-Q4_K_M.gguf",
        "download_url": "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        "size_gb": 2.32,
        "size_bytes": 2489894016,
        "min_device_ram_gb": 8,
        "context_length": 8192,
        "license": "Gemma Terms (acceptance required)",
        "notes": "Strong multilingual + summarization. First-class on Android via Google AI Edge / MediaPipe.",
        "family": "gemma3",
    },
]


def get_model(model_id: str) -> dict | None:
    for m in MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return None
