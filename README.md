# Hone ARC Miner

This workspace contains a low-cost Hone SN5 miner candidate.

## Pieces

- `solver/`: repository content that Hone validators clone and execute in their sandbox.
- `miner-server/`: tiny `/info` server to point validators at the solver repo.
- `tools/`: local evaluation scripts.
- `RUNBOOK.md`: launch checklist and tuning loop.
- `notebooks/hone_colab_llm_eval.ipynb`: Colab LLM experiment for prompt/model testing.

## Strategy

The solver has two layers:

- A deterministic low-cost layer for exact simple programs, color rules, overlays, crops, gravity, shifts, zooms, and common object heuristics.
- An optional vLLM layer for Hone validator runs. When `MINER_USE_VLLM=true`, the `/info` server requests a validator-side `1xH200` vLLM companion using `Qwen/Qwen2.5-7B-Instruct`. Our side still only needs GitHub plus a tiny `/info` server.

## Next Steps

1. Iterate locally with `python3 tools/evaluate_local.py --n 20 --seed 7`.
2. Push this repository to GitHub.
3. Deploy `miner-server/` on a cheap VPS or serverless container with `MINER_REPO_URL` set.
4. Register on SN5 once the public endpoint and repo are ready.
5. Use validator feedback to tune model, prompt, `VLLM_ATTEMPTS`, and search caps.

## Current Baseline

Local no-vLLM synthetic smoke tests compile and run, but exact-match accuracy is not yet competitive. This repo should be treated as a validator-side vLLM candidate plus an improving deterministic fallback, not a finished deterministic solver.

Latest local checks on 2026-05-13:

- Safe default: `python3 tools/evaluate_local.py --n 30 --seed 7 --chain-min 3 --chain-max 7`
  - `exact=7/30`, `shape=0.933`, `partial=0.823`, `grid=0.719`, `elapsed=97.7s`.
- Cross-seed smoke: `python3 tools/evaluate_local.py --n 20 --seed 11 --chain-min 3 --chain-max 7`
  - `exact=5/20`, `shape=0.800`, `partial=0.740`, `grid=0.649`, `elapsed=73.1s`.

Keep these deterministic solver flags off by default unless benchmarking says otherwise:

- `ARC_ENABLE_TWO_STAGE=1`: broader exact post-chain search; can find extra exacts but is slow.
- `ARC_ENABLE_SMALL_ZOOM_TARGETS=1`: solved one extra seed-7 n30 task, but raised elapsed to about 195s.
