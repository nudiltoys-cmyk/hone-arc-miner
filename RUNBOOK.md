# Hone SN5 Miner Runbook

## Cheap Launch Shape

Hone is unusually miner-friendly because validators clone and run the solver repo in their sandbox. We do not need to rent an H200 box just to serve the miner. Our side needs:

- A public GitHub repository containing `solver/`.
- A tiny public HTTP service running `miner-server/`.
- A registered SN5 miner hotkey with the public IP and port posted on-chain.

The `/info` endpoint currently requests `1xH200` with a validator-side vLLM sidecar. The model weights are downloaded in the validator prep phase and served by the validator sandbox.

## Pre-Registration Gate

Do not register until all of these are true:

1. `curl https://YOUR_HOST/health` returns `{"status":"ok"}`.
2. `curl https://YOUR_HOST/info` returns the final GitHub repo URL and hotkey.
3. The GitHub repo is public and has the required files in `solver/`.
4. We have run at least one validator-style sandbox job or a cheap local smoke test.
5. Current SN5 burn is checked and acceptable with a hard cap.

## Deploy Server

Set these environment variables on the server:

```text
MINER_REPO_URL=https://github.com/nudiltoys-cmyk/hone-arc-miner
MINER_REPO_BRANCH=main
MINER_REPO_PATH=solver
MINER_WEIGHT_CLASS=1xH200
MINER_USE_VLLM=true
MINER_HOTKEY_SS58=YOUR_HOTKEY
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_DTYPE=half
VLLM_GPU_MEMORY_UTIL=0.8
VLLM_MAX_MODEL_LEN=12000
VLLM_ATTEMPTS=1
VLLM_MAX_TOKENS=1800
```

Build and run:

```bash
docker build -t hone-arc-miner-info miner-server
docker run -d --name hone-arc-miner-info --env-file miner-server/.env -p 8091:8091 hone-arc-miner-info
```

## Register

Use a dedicated hotkey. Keep the burn capped.

```bash
btcli wallet new_hotkey --wallet.name default --wallet.hotkey hone-miner
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey hone-miner
python research-hone/tools/post_ip_chain.py --wallet-name default --hotkey hone-miner --ip YOUR_PUBLIC_IP --port 8091
```

## Tuning Loop

Once validator results arrive:

- If vLLM returns valid but wrong grids, tune prompt and `VLLM_ATTEMPTS`.
- If outputs truncate, raise `VLLM_MAX_TOKENS`.
- If vLLM cannot start, lower `VLLM_MAX_MODEL_LEN` or switch to a smaller model.
- If latency hurts, keep `VLLM_ATTEMPTS=1` and improve deterministic fast paths.

## Deterministic Solver Profiles

Use the safe default first:

```bash
python3 tools/evaluate_local.py --n 30 --seed 7 --chain-min 3 --chain-max 7 --json-out /tmp/hone_default_n30.json
```

Current safe default result from 2026-05-14:

```text
exact=9/30 (0.300) shape=0.900 partial=0.820 grid=0.729 elapsed=97.3s
```

Current cross-seed smoke from 2026-05-14:

```text
exact=15/20 (0.750) shape=0.950 partial=0.934 grid=0.903 elapsed=33.9s
```

Latest saved validator-style dry run from 2026-05-14:

```text
exact=13/20 (0.650) shape=0.900 partial=0.810 grid=0.756
```

Useful diagnostic mode:

```bash
python3 tools/evaluate_local.py --n 10 --seed 7 --chain-min 3 --chain-max 7 --include-grids --json-out /tmp/hone_debug_n10.json
```

Experimental knobs:

- `ARC_ENABLE_SMALL_ZOOM_TARGETS=1` found one extra seed-7 n30 exact (`6/30`) but took about `195s`; do not default it yet.
- `ARC_ENABLE_TWO_STAGE=1 ARC_POST_BEAM_WIDTH=8` is a broader post-chain exact search for investigation.
- `ARC_MAX_SEARCH_DEPTH=4` can improve exact rate on some samples but may reduce partial score and raise latency.
- `ARC_TARGETED_MAX_STATES` defaults to `20000`; lower values are faster but can miss row-diagonal post chains.
