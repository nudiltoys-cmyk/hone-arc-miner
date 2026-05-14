#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HONE_ROOT = REPO_ROOT.parent / "research-hone"
SANDBOX_ROOT = HONE_ROOT / "sandbox_runner"
sys.path.insert(0, str(SANDBOX_ROOT))

from storage.dataset_storage import DatasetStorage  # noqa: E402
from synthetics.arc_agi2_generator import ARC2Generator  # noqa: E402
from utils.metrics import calculate_detailed_metrics  # noqa: E402


def _generate_problem(
    generator: ARC2Generator,
    storage: DatasetStorage,
    *,
    chain_length: int,
    max_attempts: int = 200,
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


def _chain_length() -> int:
    return random.randint(3, 5) if random.random() < 0.5 else random.randint(5, 7)


def _write_dataset(
    *,
    n: int,
    seed: int,
    input_dir: Path,
    validation_path: Path,
) -> list[dict[str, Any]]:
    random.seed(seed)
    generator = ARC2Generator(seed=seed)
    storage = DatasetStorage(Path("/tmp/hone_arc_miner_validator_dry_storage"))

    miner_tasks: list[dict[str, Any]] = []
    validation_tasks: list[dict[str, Any]] = []

    for index in range(n):
        problem = _generate_problem(
            generator,
            storage,
            chain_length=_chain_length(),
        )
        miner_tasks.append(
            {
                "train_examples": problem["train_examples"],
                "test_input": problem["test_input"],
                "task_hash": problem["task_hash"],
            }
        )
        validation_tasks.append(
            {
                "index": index,
                "task_hash": problem["task_hash"],
                "expected_output": problem["test_output"],
                "metadata": problem["metadata"],
            }
        )

    input_dir.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    (input_dir / "miner_current_dataset.json").write_text(
        json.dumps(
            {
                "generated_by": "hone-arc-miner/tools/validator_dry_run.py",
                "seed": seed,
                "tasks": miner_tasks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(
            {
                "generated_by": "hone-arc-miner/tools/validator_dry_run.py",
                "seed": seed,
                "tasks": validation_tasks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return validation_tasks


def _score(output_dir: Path, validation_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    results_path = output_dir / "results.json"
    results_data = json.loads(results_path.read_text(encoding="utf-8"))

    predictions_with_expected: list[dict[str, Any]] = []
    predictions = results_data.get("predictions", [])
    validation_by_index = {item["index"]: item for item in validation_tasks}

    for fallback_index, prediction in enumerate(predictions):
        index = int(prediction.get("problem_index", fallback_index))
        validation = validation_by_index.get(index)
        if validation is None:
            continue
        prediction = dict(prediction)
        prediction["test_output"] = validation["expected_output"]
        prediction["metadata"] = validation.get("metadata", {})
        predictions_with_expected.append(prediction)

    return calculate_detailed_metrics({"predictions": predictions_with_expected})


def _summarize_misses(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    misses: list[dict[str, Any]] = []
    for row in metrics.get("per_problem", []):
        if row.get("exact_match"):
            continue
        metadata = row.get("metadata") or {}
        chain = metadata.get("transformation_chain") or []
        chain_names = [
            str(step.get("name", ""))
            for step in chain
            if isinstance(step, dict)
        ]
        misses.append(
            {
                "problem_index": row.get("problem_index"),
                "base_task": metadata.get("base_task"),
                "chain_length": metadata.get("chain_length"),
                "chain_names": chain_names,
                "partial_correctness": row.get("partial_correctness"),
                "grid_similarity": row.get("grid_similarity"),
                "shape_match": row.get("shape_match"),
                "predicted_shape": row.get("predicted_shape"),
                "expected_shape": row.get("expected_shape"),
            }
        )
    return misses


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local validator-style ARC dry run")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hone_validator_dry_run"))
    parser.add_argument(
        "--solver-dir",
        type=Path,
        default=REPO_ROOT / "solver",
        help="Directory containing arc_main.py, usually solver/ from this repo or a clean clone",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.work_dir / "input"
    output_dir = args.work_dir / "output"
    validation_path = args.work_dir / "validation_dataset.json"

    start = time.time()
    validation_tasks = _write_dataset(
        n=args.n,
        seed=args.seed,
        input_dir=input_dir,
        validation_path=validation_path,
    )

    cmd = [
        sys.executable,
        str(args.solver_dir / "arc_main.py"),
        "--phase",
        "inference",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=args.solver_dir)

    metrics = _score(output_dir, validation_tasks)
    elapsed = time.time() - start
    aggregate = metrics["aggregate"]
    report = {
        "n": args.n,
        "seed": args.seed,
        "solver_dir": str(args.solver_dir),
        "work_dir": str(args.work_dir),
        "elapsed_s": elapsed,
        "aggregate": aggregate,
        "misses": _summarize_misses(metrics),
    }

    print(
        "validator_dry_run exact={num_exact_matches}/{total_problems} "
        "({exact_match_rate:.3f}) shape={shape_match_rate:.3f} "
        "partial={avg_partial_correctness:.3f} grid={avg_grid_similarity:.3f} "
        "elapsed={elapsed:.1f}s".format(elapsed=elapsed, **aggregate)
    )
    for miss in report["misses"][:10]:
        print(
            "miss #{problem_index}: base={base_task} chain_len={chain_length} "
            "shape={predicted_shape}->{expected_shape} partial={partial_correctness:.3f} "
            "chain={chain_names}".format(**miss)
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
