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


def _load_model(model_id: str, load_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}

    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def _chat_completion(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "You solve ARC-AGI grid puzzles. Return only a JSON object "
                "with key predicted_output whose value is a rectangular grid "
                "of integers 0 through 9."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate(
    *,
    n: int,
    seed: int,
    chain_min: int,
    chain_max: int,
    model_id: str,
    load_4bit: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    random.seed(seed)
    generator = ARC2Generator(seed=seed)
    storage = DatasetStorage(Path("/tmp/hone_arc_miner_hf_eval"))
    solver = ARCSolver()
    tokenizer, model = _load_model(model_id, load_4bit)

    rows: list[dict[str, Any]] = []
    start = time.time()
    for index in range(n):
        chain_length = random.randint(chain_min, chain_max)
        problem = _generate_problem(generator, storage, chain_length=chain_length)
        prompt = solver._build_vllm_prompt(  # noqa: SLF001
            problem["train_examples"],
            problem["test_input"],
            problem["task_hash"],
        )
        raw = _chat_completion(tokenizer, model, prompt, max_new_tokens)
        predicted = solver._solve_from_llm_content(  # noqa: SLF001
            raw,
            problem["train_examples"],
            problem["test_input"],
        )
        if predicted is None:
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
                "raw_model_output": raw[:4000],
            }
        )
        print(
            "problem {}/{} exact={} partial={:.3f} shape={} -> {}".format(
                index + 1,
                n,
                metrics["exact_match"],
                metrics["partial_correctness"],
                metrics["predicted_shape"],
                metrics["expected_shape"],
            ),
            flush=True,
        )

    exact = sum(1 for row in rows if row["exact_match"])
    shape_matches = sum(1 for row in rows if row["shape_match"])
    elapsed = time.time() - start
    return {
        "n": n,
        "seed": seed,
        "chain_min": chain_min,
        "chain_max": chain_max,
        "model_id": model_id,
        "load_4bit": load_4bit,
        "exact_matches": exact,
        "exact_match_rate": exact / n if n else 0.0,
        "shape_match_rate": shape_matches / n if n else 0.0,
        "avg_partial_correctness": sum(row["partial_correctness"] for row in rows) / n if n else 0.0,
        "avg_grid_similarity": sum(row["grid_similarity"] for row in rows) / n if n else 0.0,
        "elapsed_s": elapsed,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Hone ARC miner with a Hugging Face LLM")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chain-min", type=int, default=3)
    parser.add_argument("--chain-max", type=int, default=7)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    report = evaluate(
        n=args.n,
        seed=args.seed,
        chain_min=args.chain_min,
        chain_max=args.chain_max,
        model_id=args.model_id,
        load_4bit=args.load_4bit,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        "model={model_id} exact={exact_matches}/{n} ({exact_match_rate:.3f}) "
        "shape={shape_match_rate:.3f} partial={avg_partial_correctness:.3f} "
        "grid={avg_grid_similarity:.3f} elapsed={elapsed_s:.1f}s".format(**report)
    )
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
