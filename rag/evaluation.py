"""Evaluation helpers for document/page-level RAG retrieval.

These metrics evaluate whether manually labeled relevant PDF pages appear
among retrieved results. They do not evaluate answer correctness or prove
that a generated claim is supported by its citation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_MISSING = object()


class EvaluationDataError(ValueError):
    """Raised when a retrieval evaluation dataset is invalid."""


def _get(value: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a mapping or object."""
    for name in names:
        if isinstance(value, Mapping):
            result = value.get(name, _MISSING)
        else:
            result = getattr(value, name, _MISSING)

        if result is not _MISSING and result is not None:
            return result

    return default


def _normalise_document_name(value: Any) -> str | None:
    """Compare source filenames without directory or case differences."""
    if value is None:
        return None

    name = str(value).strip().replace("\\", "/")
    if not name:
        return None

    return name.rsplit("/", 1)[-1].casefold()


def _to_positive_page_number(value: Any) -> int | None:
    """Convert a page value to a one-based positive integer."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
        if number.is_integer() and number >= 1:
            return int(number)
    except (TypeError, ValueError):
        pass

    match = re.search(r"\d+", str(value))
    if match:
        try:
            number = int(match.group(0))
        except ValueError:
            return None

        return number if number >= 1 else None

    return None


def _source_key(source: Any) -> tuple[str, int | None] | None:
    """Return (normalised PDF filename, page number) for a source."""
    metadata = _get(source, "metadata", default={})
    if not isinstance(metadata, Mapping):
        metadata = {}

    nested_source = _get(source, "chunk", "document", default=None)

    document_name = _get(
        source,
        "document_name",
        "filename",
        "source",
        default=None,
    )

    if document_name is None:
        document_name = _get(
            metadata,
            "document_name",
            "filename",
            "source",
            default=None,
        )

    if document_name is None and nested_source is not None:
        document_name = _get(
            nested_source,
            "document_name",
            "filename",
            "source",
            default=None,
        )

    normalised_name = _normalise_document_name(document_name)
    if normalised_name is None:
        return None

    raw_page = _get(
        source,
        "page_number",
        "page",
        "page_num",
        "page_no",
        default=None,
    )

    if raw_page is None:
        raw_page = _get(
            metadata,
            "page_number",
            "page",
            "page_num",
            "page_no",
            default=None,
        )

    if raw_page is None and nested_source is not None:
        raw_page = _get(
            nested_source,
            "page_number",
            "page",
            "page_num",
            "page_no",
            default=None,
        )

    page_number = _to_positive_page_number(raw_page)

    # Support zero-based page_index metadata when a one-based page number
    # is not available.
    if page_number is None:
        page_index = _get(metadata, "page_index", default=None)
        if page_index is None:
            page_index = _get(source, "page_index", default=None)

        if page_index is not None and not isinstance(page_index, bool):
            try:
                index_number = int(page_index)
                if index_number >= 0:
                    page_number = index_number + 1
            except (TypeError, ValueError):
                pass

    return normalised_name, page_number


def _validate_case(case: Any, position: int) -> dict[str, Any]:
    """Validate one dataset case and produce a compact internal record."""
    if not isinstance(case, Mapping):
        raise EvaluationDataError(
            f"Case {position} must be a JSON object/dictionary."
        )

    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        raise EvaluationDataError(
            f"Case {position} needs a non-empty 'question'."
        )

    case_id = case.get("id", f"q-{position}")
    case_id = str(case_id).strip()
    if not case_id:
        raise EvaluationDataError(
            f"Case {position} has an empty 'id'."
        )

    relevant_sources = case.get("relevant_sources")
    if (
        not isinstance(relevant_sources, list)
        or not relevant_sources
    ):
        raise EvaluationDataError(
            f"Case {case_id!r} needs a non-empty 'relevant_sources' list."
        )

    relevant_keys: list[tuple[str, int]] = []

    for source_position, source in enumerate(relevant_sources, start=1):
        source_key = _source_key(source)
        if source_key is None:
            raise EvaluationDataError(
                f"Case {case_id!r}, relevant source {source_position}: "
                "include a valid 'document_name'."
            )

        document_name, page_number = source_key
        if page_number is None:
            raise EvaluationDataError(
                f"Case {case_id!r}, relevant source {source_position}: "
                "include a one-based 'page_number'."
            )

        key = (document_name, page_number)
        if key not in relevant_keys:
            relevant_keys.append(key)

    language = case.get("language", "unspecified")
    if language is None or not str(language).strip():
        language = "unspecified"

    return {
        "id": case_id,
        "language": str(language),
        "question": question.strip(),
        "relevant_keys": relevant_keys,
    }


def load_evaluation_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate UTF-8 JSON Lines evaluation data."""
    file_path = Path(path)
    if not file_path.is_file():
        raise EvaluationDataError(
            f"Evaluation file not found: {file_path}"
        )

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        file_handle = file_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise EvaluationDataError(
            f"Could not open evaluation file: {file_path}"
        ) from exc

    with file_handle:
        for line_number, raw_line in enumerate(file_handle, start=1):
            line = raw_line.strip()

            # Blank lines and comment lines are ignored.
            if not line or line.startswith("#"):
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(
                    f"{file_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            try:
                validated = _validate_case(row, len(cases) + 1)
            except EvaluationDataError as exc:
                raise EvaluationDataError(
                    f"{file_path}:{line_number}: {exc}"
                ) from exc

            id_key = validated["id"].casefold()
            if id_key in seen_ids:
                raise EvaluationDataError(
                    f"{file_path}:{line_number}: duplicate case ID "
                    f"{validated['id']!r}."
                )

            seen_ids.add(id_key)
            cases.append(validated)

    if not cases:
        raise EvaluationDataError(
            f"No evaluation questions found in {file_path}."
        )

    return cases


def _as_result_list(result: Any) -> list[Any]:
    """Convert common retrieval outputs to a list."""
    if result is None:
        return []

    if isinstance(result, Mapping):
        for name in ("results", "matches", "hits", "items"):
            value = result.get(name)
            if isinstance(value, (list, tuple)):
                return list(value)
        return [result]

    if isinstance(result, (str, bytes)):
        raise TypeError(
            "pipeline.retrieve() must return source records, not a string."
        )

    try:
        return list(result)
    except TypeError as exc:
        raise TypeError(
            "pipeline.retrieve() must return an iterable of source records."
        ) from exc


def evaluate_retrieval(
    pipeline: Any,
    cases: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate page-level retrieval against manually labeled sources.

    The pipeline must expose:
        pipeline.retrieve(question, top_k=top_k)

    Metric units are unique PDF pages, not chunks. If multiple retrieved
    chunks come from the same page, that page counts once.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if pipeline is None or not callable(getattr(pipeline, "retrieve", None)):
        raise ValueError(
            "pipeline must provide a callable retrieve(question, top_k=...) method."
        )

    if not isinstance(cases, list) or not cases:
        raise EvaluationDataError(
            "cases must be a non-empty list of evaluation questions."
        )

    validated_cases = [
        _validate_case(case, position)
        for position, case in enumerate(cases, start=1)
    ]

    retrieve = pipeline.retrieve
    per_question: list[dict[str, Any]] = []
    missing_page_metadata_total = 0
    missing_document_metadata_total = 0

    for case in validated_cases:
        try:
            raw_hits = retrieve(case["question"], top_k=top_k)
        except Exception as exc:
            raise RuntimeError(
                f"Retrieval failed for evaluation case {case['id']!r}."
            ) from exc

        hits = _as_result_list(raw_hits)[:top_k]
        ranked_pages: list[tuple[str, int | None]] = []
        seen_pages: set[tuple[str, int | None]] = set()
        missing_page_count = 0
        missing_document_count = 0

        for hit in hits:
            key = _source_key(hit)

            if key is None:
                missing_document_count += 1
                continue

            document_name, page_number = key

            if page_number is None:
                missing_page_count += 1

            page_key = (document_name, page_number)
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                ranked_pages.append(page_key)

        missing_page_metadata_total += missing_page_count
        missing_document_metadata_total += missing_document_count

        relevant_set = set(case["relevant_keys"])

        matched_relevant_pages = {
            key for key in ranked_pages if key in relevant_set
        }

        if ranked_pages:
            precision_at_k = len(matched_relevant_pages) / len(ranked_pages)
        else:
            precision_at_k = 0.0

        recall_at_k = len(matched_relevant_pages) / len(relevant_set)

        first_relevant_rank = next(
            (
                rank
                for rank, key in enumerate(ranked_pages, start=1)
                if key in relevant_set
            ),
            None,
        )

        reciprocal_rank_at_k = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        per_question.append(
            {
                "id": case["id"],
                "language": case["language"],
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "reciprocal_rank_at_k": reciprocal_rank_at_k,
                "hit_at_k": first_relevant_rank is not None,
                "relevant_page_count": len(relevant_set),
                "retrieved_unique_page_count": len(ranked_pages),
                "missing_page_metadata_count": missing_page_count,
                "missing_document_metadata_count": missing_document_count,
            }
        )

    count = len(per_question)

    def mean(metric_name: str) -> float:
        return sum(
            float(result[metric_name])
            for result in per_question
        ) / count

    return {
        "evaluation_cases": count,
        "top_k": top_k,
        "unit": "unique document-page pairs",
        "precision_at_k": mean("precision_at_k"),
        "recall_at_k": mean("recall_at_k"),
        "mrr_at_k": mean("reciprocal_rank_at_k"),
        "hit_rate_at_k": mean("hit_at_k"),
        "missing_page_metadata_total": missing_page_metadata_total,
        "missing_document_metadata_total": missing_document_metadata_total,
        "per_question": per_question,
    }