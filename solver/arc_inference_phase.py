from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from arc_solver import ARCSolver
from arc_utils import load_input_data, save_output_data


def run_inference_phase(input_dir: Path, output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("INFERENCE PHASE - hone-arc-miner")
    print("=" * 60)

    data = load_input_data(input_dir)
    problems: list[dict[str, Any]] = data["tasks"]
    solver = ARCSolver()
    predictions: list[dict[str, Any]] = []

    for i, problem in enumerate(problems):
        try:
            predicted_output = solver.solve(
                train_examples=problem["train_examples"],
                test_input=problem["test_input"],
                task_hash=problem.get("task_hash"),
            )
            predictions.append(
                {
                    "problem_index": i,
                    "task_hash": problem.get("task_hash"),
                    "predicted_output": predicted_output,
                }
            )
            print(f"problem {i + 1}/{len(problems)} -> {len(predicted_output)}x{len(predicted_output[0])}")
        except Exception as exc:
            print(f"problem {i + 1}/{len(problems)} failed: {exc}")
            predictions.append(
                {
                    "problem_index": i,
                    "task_hash": problem.get("task_hash"),
                    "predicted_output": None,
                    "error": str(exc),
                }
            )

    save_output_data(
        {
            "phase": "inference",
            "status": "success",
            "num_problems_solved": sum(p.get("predicted_output") is not None for p in predictions),
            "predictions": predictions,
        },
        output_dir,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Hone ARC inference phase")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run_inference_phase(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
