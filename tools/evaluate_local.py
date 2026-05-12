#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
HONE_ROOT = REPO_ROOT.parent / "research-hone"
SANDBOX_ROOT = HONE_ROOT / "sandbox_runner"
sys.path.insert(0, str(REPO_ROOT / "solver"))
sys.path.insert(0, str(SANDBOX_ROOT))

from arc_solver import ARCSolver  # noqa: E402
from storage.dataset_storage import DatasetStorage  # noqa: E402
from synthetics.arc_agi2_generator import ARC2Generator  # noqa: E402
from utils.metrics import calculate_metrics_for_prediction  # noqa: E402


def _generate_problem(
    generator: ARC2Generator,
    storage: DatasetStorage,
    *,
    chain_length: int,
    max_attempts: int = 100,
) -> dict[str, Any]:
    for _ in range(max_attempts):
        try:
            problem = generator.generate_problem_set(
                num_train=3,
                num_test=1,
                chain_length=chain_length,
                preserves_size_only=False,
            )
            if len(problem["train_examples"]) >= 3:
                return {
                    "train_examples": problem["train_examples"],
                    "test_input": problem["test_input"],
                    "test_output": problem["test_output"],
                    "task_hash": storage.hash_task(problem),
                    "metadata": problem.get("metadata", {}),
                }
        except Exception:
            continue
    raise RuntimeError("could not generate a valid problem")


def evaluate(n: int, seed: int, chain_min: int, chain_max: int) -> dict[str, Any]:
    random.seed(seed)
    generator = ARC2Generator(seed=seed)
    storage = DatasetStorage(Path("/tmp/hone_arc_miner_eval"))
    solver = ARCSolver()

    rows: list[dict[str, Any]] = []
    start = time.time()
    for index in range(n):
        chain_length = random.randint(chain_min, chain_max)
        problem = _generate_problem(generator, storage, chain_length=chain_length)
        predicted = solver.solve(
            problem["train_examples"],
            problem["test_input"],
            task_hash=problem["task_hash"],
        )
        metrics = calculate_metrics_for_prediction(
            predicted,
            problem["test_output"],
            problem.get("metadata", {}),
        )
        rows.append(
            {
                "index": index,
                "task_hash": problem["task_hash"],
                "base_task": problem["metadata"].get("base_task"),
                "chain": problem["metadata"].get("transformation_chain"),
                "chain_length": problem["metadata"].get("chain_length"),
                "exact_match": metrics["exact_match"],
                "partial_correctness": metrics["partial_correctness"],
                "grid_similarity": metrics["grid_similarity"],
                "shape_match": metrics["shape_match"],
                "predicted_shape": metrics["predicted_shape"],
                "expected_shape": metrics["expected_shape"],
            }
        )

    exact = sum(1 for row in rows if row["exact_match"])
    shape_matches = sum(1 for row in rows if row["shape_match"])
    elapsed = time.time() - start
    return {
        "n": n,
        "seed": seed,
        "chain_min": chain_min,
        "chain_max": chain_max,
        "exact_matches": exact,
        "exact_match_rate": exact / n if n else 0.0,
        "shape_match_rate": shape_matches / n if n else 0.0,
        "avg_partial_correctness": sum(row["partial_correctness"] for row in rows) / n if n else 0.0,
        "avg_grid_similarity": sum(row["grid_similarity"] for row in rows) / n if n else 0.0,
        "elapsed_s": elapsed,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hone-arc-miner locally")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chain-min", type=int, default=3)
    parser.add_argument("--chain-max", type=int, default=7)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    report = evaluate(args.n, args.seed, args.chain_min, args.chain_max)
    print(
        "exact={exact_matches}/{n} ({exact_match_rate:.3f}) "
        "shape={shape_match_rate:.3f} partial={avg_partial_correctness:.3f} "
        "grid={avg_grid_similarity:.3f} elapsed={elapsed_s:.1f}s".format(**report)
    )
    misses = [row for row in report["rows"] if not row["exact_match"]][:10]
    for row in misses:
        print(
            "miss #{index}: base={base_task} chain_len={chain_length} "
            "shape={predicted_shape}->{expected_shape} partial={partial_correctness:.3f} "
            "chain={chain}".format(**row)
        )

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
