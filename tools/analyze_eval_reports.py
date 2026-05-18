#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _miss_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "rows" in report:
        rows = [row for row in report["rows"] if not row.get("exact_match")]
        return [
            {
                "index": row.get("index"),
                "base_task": row.get("base_task"),
                "chain_names": row.get("chain_names") or [],
                "partial_correctness": row.get("partial_correctness"),
                "grid_similarity": row.get("grid_similarity"),
                "shape_match": row.get("shape_match"),
                "predicted_shape": row.get("predicted_shape"),
                "expected_shape": row.get("expected_shape"),
            }
            for row in rows
        ]
    return report.get("misses", [])


def _score_line(path: Path, report: dict[str, Any]) -> str:
    if "exact_match_rate" in report:
        return (
            f"{path.name}: exact={report.get('exact_matches')}/{report.get('n')} "
            f"({report.get('exact_match_rate', 0):.3f}) "
            f"shape={report.get('shape_match_rate', 0):.3f} "
            f"partial={report.get('avg_partial_correctness', 0):.3f}"
        )
    aggregate = report.get("aggregate", {})
    return (
        f"{path.name}: exact={aggregate.get('num_exact_matches')}/{aggregate.get('total_problems')} "
        f"({aggregate.get('exact_match_rate', 0):.3f}) "
        f"shape={aggregate.get('shape_match_rate', 0):.3f} "
        f"partial={aggregate.get('avg_partial_correctness', 0):.3f}"
    )


def _shape_key(row: dict[str, Any]) -> str:
    return "shape_ok" if row.get("shape_match") else "shape_bad"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank ARC misses across local eval JSON reports")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    base_counts: Counter[Any] = Counter()
    first_counts: Counter[str] = Counter()
    last_counts: Counter[str] = Counter()
    shape_counts: Counter[tuple[Any, str]] = Counter()
    partials: dict[Any, list[float]] = defaultdict(list)
    examples: dict[Any, list[str]] = defaultdict(list)

    print("Reports")
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        print(f"- {_score_line(path, report)}")
        for row in _miss_rows(report):
            base = row.get("base_task")
            chain = row.get("chain_names") or []
            base_counts[base] += 1
            if chain:
                first_counts[chain[0]] += 1
                last_counts[chain[-1]] += 1
            shape_counts[(base, _shape_key(row))] += 1
            partial = row.get("partial_correctness")
            if isinstance(partial, (int, float)):
                partials[base].append(float(partial))
            if len(examples[base]) < 3:
                examples[base].append(
                    "idx={idx} shape={pred}->{exp} partial={partial} chain={chain}".format(
                        idx=row.get("index", row.get("problem_index")),
                        pred=row.get("predicted_shape"),
                        exp=row.get("expected_shape"),
                        partial=round(float(partial), 3) if isinstance(partial, (int, float)) else partial,
                        chain=",".join(chain),
                    )
                )

    print("\nTop Missed Base Tasks")
    for base, count in base_counts.most_common(args.top):
        vals = partials.get(base, [])
        avg_partial = sum(vals) / len(vals) if vals else 0.0
        shape_ok = shape_counts.get((base, "shape_ok"), 0)
        shape_bad = shape_counts.get((base, "shape_bad"), 0)
        print(f"- base={base} count={count} shape_ok={shape_ok} shape_bad={shape_bad} avg_partial={avg_partial:.3f}")
        for example in examples.get(base, [])[:2]:
            print(f"  {example}")

    print("\nTop First Transforms")
    for name, count in first_counts.most_common(args.top):
        print(f"- {name}: {count}")

    print("\nTop Last Transforms")
    for name, count in last_counts.most_common(args.top):
        print(f"- {name}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
