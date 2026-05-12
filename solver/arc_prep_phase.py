from __future__ import annotations

import os
from pathlib import Path


def run_prep_phase(cache_dir: Path | None = None) -> None:
    """Download optional vLLM assets during the network-enabled prep phase."""
    if os.getenv("ENABLE_VLLM_PREP", "0").lower() not in {"1", "true", "yes"}:
        print("PREP PHASE - no external assets required")
        return

    from huggingface_hub import snapshot_download

    cache_dir = cache_dir or Path(os.getenv("MODEL_CACHE_DIR", "/app/models"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    local_dir = cache_dir / model_name.replace("/", "--")
    print(f"PREP PHASE - downloading {model_name} to {local_dir}")

    if local_dir.exists() and any(local_dir.iterdir()):
        print("model cache already populated; skipping download")
        return

    snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_dir),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print("model download complete")
