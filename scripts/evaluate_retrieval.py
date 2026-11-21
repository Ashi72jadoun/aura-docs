"""Run page-level retrieval evaluation against labeled questions."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from rag.evaluation import evaluate_retrieval, load_evaluation_cases


def load_pipeline_factory(factory_path: str) -> Any:
    """Load a no-argument factory written as 'module.path:function_name'."""
    if ":" not in factory_path:
        raise ValueError(
            "Factory must use the format 'python.module:function_name'."
        )

    module_name, function_name = factory_path.split(":", 1)
    if not module_name.strip() or not function_name.strip():
        raise ValueError(
            "Factory must use the format 'python.module:function_name'."
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)

    if not callable(factory):
        raise TypeError(f"{factory_path!r} is not callable.")

    return factory


def convert_loaded_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert loader output into the case format expected by the evaluator."""
    converted: list[dict[str, Any]] = []

    for case in cases:
        relevant_sources = []

        for document_name, page_number in case["relevant_keys"]:
            relevant_sources.append(
                {
                    "document_name": document_name,
                    "page_number": page_number,
                }
            )

        converted.append(
            {
                "id": case["id"],
                "language": case["language"],
                "question": case["question"],
                "relevant_sources": relevant_sources,
            }
        )

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PDF retrieval against labeled document pages."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a UTF-8 JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--factory",
        required=True,
        help="Pipeline factory, e.g. rag.demo_factory:create_pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to evaluate (default: 5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for a local JSON results file.",
    )

    args = parser.parse_args()

    try:
        loaded_cases = load_evaluation_cases(args.data)
        cases = convert_loaded_cases(loaded_cases)

        factory = load_pipeline_factory(args.factory)
        pipeline = factory()

        result = evaluate_retrieval(
            pipeline=pipeline,
            cases=cases,
            top_k=args.top_k,
        )
    except Exception as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
        print(f"\nSaved local results to: {output_path}")


if __name__ == "__main__":
    main()