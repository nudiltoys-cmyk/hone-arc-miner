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
2. Recheck with `python3 tools/validator_dry_run.py --n 100 --seed 20260515 --solver-dir /tmp/hone_miner_clone_648684d/solver`.
3. Push this repository to GitHub once the validator-style dry run is comfortably over floor.
4. Deploy `miner-server/` on a cheap VPS or serverless container with `MINER_REPO_URL` set.
5. Register on SN5 only after the clean-clone dry run clears the 20% floor with margin and current burn is acceptable.
6. Use validator feedback to tune model, prompt, `VLLM_ATTEMPTS`, and search caps.

## Current Baseline

Local no-vLLM synthetic smoke tests compile and run. The deterministic layer has strong pockets, but the proper clean-clone validator-format run is still below Hone's 20% exact-rate floor. Treat this as an improving miner candidate, not a launch-ready solver.

Latest local checks on 2026-05-14:

- Safe default: `python3 tools/evaluate_local.py --n 30 --seed 7 --chain-min 3 --chain-max 7`
  - `exact=12/30`, `shape=0.933`, `partial=0.854`, `grid=0.777`, `elapsed=75.9s`.
- Cross-seed smoke: `python3 tools/evaluate_local.py --n 20 --seed 11 --chain-min 3 --chain-max 7`
  - `exact=20/20`, `shape=1.000`, `partial=1.000`, `grid=1.000`, `elapsed=16.2s`.
- Saved validator-style dry run: `/tmp/hone_validator_dry_input`
  - `exact=13/20`, `shape=0.900`, `partial=0.810`, `grid=0.756`.
- Broad multi-seed generated check:
  - `exact=46/120 (0.383)`, `shape=0.842`, `partial=0.778`, `grid=0.709`, `elapsed=898.4s`.
- Clean-clone validator-format run:
  - `python3 tools/validator_dry_run.py --n 100 --seed 20260515 --solver-dir /tmp/hone_miner_clone_648684d/solver`
  - `exact=11/100 (0.110)`, `shape=0.780`, `partial=0.706`, `grid=0.608`, `elapsed=762.5s`.

Launch gate from 2026-05-14: failed. Do not register/burn until the clean-clone validator-format run clears 20% with margin across multiple seeds.

Keep these deterministic solver flags off by default unless benchmarking says otherwise:

- `ARC_ENABLE_TWO_STAGE=1`: broader exact post-chain search; can find extra exacts but is slow.
- `ARC_ENABLE_SMALL_ZOOM_TARGETS=1`: solved one extra seed-7 n30 task, but raised elapsed to about 195s.
