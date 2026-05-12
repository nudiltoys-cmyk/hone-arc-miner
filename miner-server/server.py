from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

app = FastAPI(title="hone-arc-miner-info", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    use_vllm = os.getenv("MINER_USE_VLLM", "true").lower() == "true"
    payload: dict[str, Any] = {
        "repo_url": os.environ["MINER_REPO_URL"],
        "repo_branch": os.getenv("MINER_REPO_BRANCH", "main"),
        "repo_commit": os.getenv("MINER_REPO_COMMIT") or None,
        "repo_path": os.getenv("MINER_REPO_PATH", "solver"),
        "weight_class": os.getenv("MINER_WEIGHT_CLASS", "1xH200"),
        "use_vllm": use_vllm,
        "custom_env_vars": _custom_env_vars(use_vllm),
        "version": os.getenv("MINER_VERSION", "0.1.0"),
        "hotkey": os.getenv("MINER_HOTKEY_SS58", ""),
    }
    if use_vllm:
        payload["vllm_config"] = {
            "model": os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "dtype": os.getenv("VLLM_DTYPE", "half"),
            "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.8")),
            "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "12000")),
        }
    return {key: value for key, value in payload.items() if value is not None}


def _custom_env_vars(use_vllm: bool) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if use_vllm:
        env_vars["ENABLE_VLLM_PREP"] = "1"
        env_vars["VLLM_MODEL"] = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        env_vars["VLLM_ATTEMPTS"] = os.getenv("VLLM_ATTEMPTS", "1")
        env_vars["VLLM_MAX_TOKENS"] = os.getenv("VLLM_MAX_TOKENS", "1800")

    custom = os.getenv("MINER_CUSTOM_ENV_VARS", "")
    for pair in custom.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            env_vars[key.strip()] = value.strip()
    return env_vars
