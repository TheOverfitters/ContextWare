from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from acf_chunker import plan_chunks
from acf_io import (
    build_output_paths,
    get_run_timestamp,
    load_categories,
    load_label_descriptions,
    load_maintenance_descriptions,
    load_ollama_api_key,
    load_processed_ids,
    load_repo_diffs,
    truncate_text,
)
from acf_parsing import (
    aggregate_chunk_vectors,
    maintenance_vector_from_payload,
    parse_multilabel_response,
    # signals_fallback_vector,
)
from acf_prompt import (
    DEFAULT_CATEGORIES,
    DEFAULT_LEAF_TO_BASE,
    DEFAULT_LEAF_TO_CLASS,
    DEFAULT_MAINTENANCE_TYPES,
    DEFAULT_MODEL,
    # SIGNAL_CATEGORY_MAP,
    # SIGNAL_PATTERNS,
    build_chat_messages,
    # build_signals,
    build_user_prompt,
    get_system_prompt,
)

# # Reverse map: category name → list of regex patterns.
# _CATEGORY_PATTERNS: dict[str, list[str]] = {}
# for _sig, _pat in SIGNAL_PATTERNS:
#     _cat = SIGNAL_CATEGORY_MAP.get(_sig)
#     if _cat:
#         _CATEGORY_PATTERNS.setdefault(_cat, []).append(_pat)
_CATEGORY_PATTERNS: dict[str, list[str]] = {}
from ollama_client import SlidingRateLimiter, call_chat_with_retries, fetch_model_context_size


# Default output dir lives OUTSIDE OneDrive-synced folders: OneDrive locks files
# mid-write while syncing, causing intermittent PermissionError on append+flush.
# LOCALAPPDATA (C:\Users\<user>\AppData\Local) is never synced; fall back to ~ elsewhere.
_DEFAULT_OUTPUT_DIR = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "acf-outputs"


# Parsing, classification, and chunking logic are all intertwined in this file since
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ACF multi-label LLM classifier over one or more "
            "repositories of diffs."
        )
    )
    # Input and output paths
    parser.add_argument(
        "--diffs-json",
        type=Path,
        default=Path("acf_diffs.json"),
        help="Path to a diffs JSON file, a JSONL file, or a directory of JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            "Where to write primary/eval JSONL. Defaults to a non-OneDrive "
            f"local path ({_DEFAULT_OUTPUT_DIR}) to avoid sync-lock write errors."
        ),
    )
    parser.add_argument(
        "--label-descriptions",
        type=Path,
        default=Path("categories.json"),
        help="Path to categories.json with category names and descriptions.",
    )
    parser.add_argument(
        "--maintenance-descriptions",
        type=Path,
        default=Path("modification_request.json"),
        help=(
            "Path to modification_request.json with the maintenance-type "
            "taxonomy (the 'why' axis, ISO/IEC/IEEE 14764)."
        ),
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated subset of categories to score (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N diffs per repository.",
    )
    parser.add_argument(
        "--repo-filter",
        type=str,
        default=None,
        help="Only process repositories whose label matches this substring.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip diffs already in output")
    parser.add_argument(
        "--include-patch",
        action="store_true",
        help="Include the (possibly truncated) diff text in output records.",
    )
    parser.add_argument(
        "--include-chunk-details",
        action="store_true",
        help="Include per-chunk classifications in the primary record.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the rendered system+user prompt for the first diff.",
    )

    # LLM Parameters and prompt engineering flags
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Override output timestamp (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the Ollama API",
    )
    parser.add_argument("--max-diff-chars", type=int, default=60000)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument(
        "--use-format-schema",
        action="store_true",
        dest="use_format_schema",
        help=(
            "Enable Ollama grammar-constrained JSON schema on the first attempt. "
            "Disabled by default: some models produce empty or truncated responses "
            "when the grammar engine struggles with complex schemas. "
            "Retries never use the schema regardless of this flag."
        ),
    )
    # Legacy alias kept for backwards compatibility.
    parser.add_argument(
        "--no-format-schema",
        action="store_false",
        dest="use_format_schema",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help=(
            "Override the model context window (Ollama num_ctx). "
            "Leave unset to use the model's own default (recommended). "
            "Set only to extend context on models that load with a small default "
            "(e.g. --context-size 8192 for models defaulting to 2048)."
        ),
    )
    parser.add_argument(
        "--retry-on-zero-flags",
        action="store_true",
        help=(
            "Re-run the entire diff when the final classification has no flagged "
            "categories (flagged=0). Uses --retry-on-invalid as the retry count. "
            "Useful when the model returns valid JSON but assigns 0.0 to all categories."
        ),
    )
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help=(
            "Disable assistant-prefill for the first attempt. "
            "By default a partial '{' is injected as the last assistant message so "
            "thinking/reasoning models produce visible JSON output instead of consuming "
            "all tokens internally. Retries always use prefill regardless of this flag."
        ),
    )
    parser.add_argument(
        "--debug-raw-response",
        action="store_true",
        help="Print the first 400 chars of the raw model response whenever invalid_json is raised.",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--retry-per-prompt", type=int, default=3)

    # Rate limiting parameters
    parser.add_argument("--requests-per-minute", type=int, default=600)
    parser.add_argument("--tokens-per-minute", type=int, default=10_000_000)
    parser.add_argument("--min-request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--rate-limit-wait-base-seconds", type=int, default=12)
    parser.add_argument("--rate-limit-wait-max-seconds", type=int, default=45)
    parser.add_argument("--hard-cooldown-seconds", type=int, default=90)
    parser.add_argument(
        "--retry-on-invalid",
        type=int,
        default=1,
        help="Retry the same prompt when JSON or schema validation fails.",
    )

    # Chunking and aggregation parameters
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=400,
        help="Number of trailing characters to repeat at the start of the next chunk.",
    )
    parser.add_argument(
        "--chunk-aggregation",
        type=str,
        default="max",
        choices=("max", "mean", "any"),
        help="How to combine per-chunk score vectors into a single per-diff vector.",
    )
    parser.add_argument(
        "--per-category-threshold",
        type=float,
        default=0.50,
        help="Score threshold used to flag a category as 'present'.",
    )
    parser.add_argument(
        "--diff-retry-on-missing-list",
        type=int,
        default=3,
        help=(
            "Re-run the same diff up to N times when the model response "
            "triggers the 'categories_missing_or_not_list' validation error. "
            "Set to 0 to disable."
        ),
    )

    return parser.parse_args()


# Input handling and chunking logic are intertwined since the chunking plan 
def resolve_input_paths(diffs_path: Path) -> list[Path]:
    """Resolve --diffs-json into a list of JSON or JSONL files to iterate."""
    if not diffs_path.exists():
        raise FileNotFoundError(f"Diffs input not found: {diffs_path}")
    if diffs_path.is_dir():
        candidates = sorted(
            p for p in diffs_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
        )
        if not candidates:
            raise ValueError(f"No JSON/JSONL files found in {diffs_path}")
        return candidates
    if diffs_path.suffix.lower() == ".jsonl":
        return [diffs_path]
    return [diffs_path]


# Prompt aggregation and response parsing logic are intertwined since we need to track which chunks produced which errors and scores in order to make informed decisions about retries and final aggregation.
def _aggregate_validation_errors(chunk_records: list[dict]) -> list[str]:
    """Collect a unique, sorted list of validation errors across all chunks.

    Errors are tagged with the chunk index in the form
    ``"chunk<i>:<error_code>"`` so the diff-retry logic in ``main`` can
    tell *which* chunk failed even when several share an error code.
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunk_records:
        idx = chunk.get("index", 0)
        for err in chunk.get("validation_errors") or []:
            if not isinstance(err, str):
                continue
            tag = f"chunk{idx}:{err}"
            if tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
    return sorted(out)


# Pre and post-processing logic are intertwined since the rationale extraction and summary selection logic depend on the per-chunk parsed results and scores.
def _build_response_schema(categories: list[str], maintenance_types: list[str]) -> dict:
    """Build the JSON schema passed to Ollama as ``format``."""
    return {
        "type": "object",
        "required": [
            "categories",
            "primary_category",
            "maintenance_types",
            "primary_maintenance_type",
            "summary",
        ],
        "properties": {
            "categories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "score", "rationale"],
                    "properties": {
                        "name":      {"type": "string", "enum": categories},
                        "score":     {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            },
            "primary_category": {"type": "string", "enum": categories},
            "maintenance_types": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "score", "rationale"],
                    "properties": {
                        "name":      {"type": "string", "enum": maintenance_types},
                        "score":     {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                },
            },
            "primary_maintenance_type": {"type": "string", "enum": maintenance_types},
            "summary":          {"type": "string"},
        },
    }

# Exstracting rationales from the highest-scoring chunk for each category is intertwined with the classification logic since we need to know which chunk had the highest score for each category in order to pull out the corresponding rationale text.
def _extract_rationale_map(
    chunk_records: list[dict],
    categories: list[str],
    key: str = "categories",
) -> dict[str, str]:
    """For each label under *key*, return the rationale from its top-scoring chunk."""
    best_scores: dict[str, float] = {c: -1.0 for c in categories}
    rationale_map: dict[str, str] = {}
    for chunk in chunk_records:
        parsed = chunk.get("parsed")
        if not isinstance(parsed, dict):
            continue
        for entry in (parsed.get(key) or []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if name not in best_scores:
                continue
            score = entry.get("score")
            if not isinstance(score, (int, float)):
                continue
            if float(score) > best_scores[name]:
                best_scores[name] = float(score)
                rationale = entry.get("rationale", "")
                if isinstance(rationale, str) and rationale.strip():
                    rationale_map[name] = rationale.strip()
    return rationale_map


def _best_summary(chunk_records: list[dict], primary_category: str) -> str:
    """Pick the most representative summary across chunks.

    Prefers the chunk whose primary_category matches the aggregated result.
    Falls back to the chunk with the highest max score, then to chunk[0].
    """
    fallback_summary = ""
    best_score = -1.0
    for chunk in chunk_records:
        parsed = chunk.get("parsed")
        if not isinstance(parsed, dict):
            continue
        summary = parsed.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            continue
        if parsed.get("primary_category") == primary_category:
            return summary.strip()
        score = max((chunk.get("vector") or {}).values(), default=0.0)
        if score > best_score:
            best_score = score
            fallback_summary = summary.strip()
    return fallback_summary


# Prefill logic for thinking/reasoning models is intertwined with the classification logic since we need to apply the prefill on the initial attempt but not on retries, and we need to reconstruct the full JSON response when the model returns only the continuation after the prefill.
_PREFILL_CHAR = "{"


def _apply_prefill(
    messages: list[dict[str, str]],
    use_prefill: bool,
) -> list[dict[str, str]]:
    """Return messages with a partial assistant message appended when *use_prefill* is True.

    The assistant prefill ``{`` forces thinking/reasoning models to write their
    JSON answer directly into the visible content rather than consuming all
    tokens in an internal reasoning phase that Ollama does not return.
    """
    if not use_prefill:
        return messages
    return messages + [{"role": "assistant", "content": _PREFILL_CHAR}]


def _reconstruct_prefilled(content: str, use_prefill: bool) -> str:
    """Restore the full JSON object when the model returned only the continuation.

    Ollama may return either:
      (a) just the continuation after the prefill (``"categories":[...],...}``)
      (b) the full content including the prefill (``{"categories":[...],...}``)

    In case (a) we prepend ``{`` so downstream parsing sees a valid JSON object.
    In case (b) the content already starts with ``{`` and is returned unchanged.
    """
    if not use_prefill:
        return content
    if content.lstrip().startswith("{"):
        return content
    return _PREFILL_CHAR + content


def classify_diff(
    *,
    diff: dict,
    args: argparse.Namespace,
    categories: list[str],
    maintenance_types: list[str],
    leaf_to_class: dict[str, str],
    leaf_to_base: dict[str, str],
    system_prompt: str,
    api_key: str,
    rate_limiter: SlidingRateLimiter,
    temperature_override: float | None = None,
) -> dict:
    """Run the multi-label classifier over a single diff (potentially chunked).

    Returns a dict ready to be written to the primary JSONL record.
    """
    diff_id = str(diff.get("diff_id", ""))
    diff_text = str(diff.get("diff_text", ""))
    commit_hash = str(diff.get("commit_hash", ""))
    commit_message = str(diff.get("commit_message", ""))
    commit_date = str(diff.get("timestamp", ""))
    filename = str(diff.get("filename", ""))

    # added_signals, removed_signals = build_signals(diff_text, filename, commit_message)
    # signals = list(dict.fromkeys(added_signals + removed_signals))
    plan = plan_chunks(
        diff_text,
        max_chars=args.max_diff_chars,
        overlap_chars=args.chunk_overlap_chars,
    )

    def _chunk_changed_lines(text: str) -> int:
        return sum(
            1 for line in text.splitlines()
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith("+++") and not line.startswith("---")
        )

    response_schema = _build_response_schema(categories, maintenance_types)
    chunk_records: list[dict] = []
    chunk_vectors: list[dict[str, float] | None] = []
    chunk_maint_vectors: list[dict[str, float] | None] = []

    for chunk in plan.chunks:
        chunk_changed = _chunk_changed_lines(chunk.text)
        user_prompt = build_user_prompt(
            chunk_text=chunk.text,
            filename=filename,
            commit_message=commit_message,
            commit_date=commit_date,
            chunk_index=chunk.index,
            chunk_total=chunk.total,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            # signals=signals,
        )
        base_messages = build_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Use grammar-constrained schema on the first attempt unless disabled.
        # Retries always go without schema: empty/truncated responses are caused
        # by the grammar engine struggling with complex schemas, and the prompt
        # instructions alone are sufficient for a well-guided retry.
        first_schema = response_schema if args.use_format_schema else None

        # Prefill: inject a partial assistant message so thinking/reasoning models
        # are forced to produce visible JSON output instead of consuming all tokens
        # in an internal reasoning phase with empty content.
        # Retries always use prefill even when --no-prefill is set (first attempt only).
        first_prefill = not getattr(args, "no_prefill", False)
        messages = _apply_prefill(base_messages, use_prefill=first_prefill)

        chunk_start = time.time()
        retry_attempted = False
        base_temp = temperature_override if temperature_override is not None else args.temperature
        result = call_chat_with_retries(
            base_url=args.ollama_base_url,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            messages=messages,
            model=args.model,
            temperature=base_temp,
            top_p=args.top_p,
            max_completion_tokens=args.max_tokens,
            rate_limiter=rate_limiter,
            retry_per_prompt=args.retry_per_prompt,
            rate_limit_wait_base_seconds=args.rate_limit_wait_base_seconds,
            rate_limit_wait_max_seconds=args.rate_limit_wait_max_seconds,
            hard_cooldown_seconds=args.hard_cooldown_seconds,
            request_logprobs=False,
            response_schema=first_schema,
            context_size=args.context_size,
        )
        raw = _reconstruct_prefilled(result.content, use_prefill=first_prefill)

        parsed, errors, vector = parse_multilabel_response(raw, categories)

        # Track the best parsed result and vector so we can use them for final aggregation even when retries yield degraded results (e.g. empty or all-zero vector) due to the grammar engine struggling with complex schemas.
        best_parsed = parsed
        best_vector = vector
        best_vector_sum = sum(vector.values())
        any_attempt_parsed = parsed is not None

        
        def _has_usable_signal(v: dict[str, float]) -> bool:
            return any(s >= args.per_category_threshold for s in v.values())

        retry_count = 0
        while (
            (parsed is None or errors or (chunk_changed > 0 and not any(v > 0.0 for v in vector.values())))
            and not _has_usable_signal(vector)
            and retry_count < args.retry_on_invalid
        ):
            retry_count += 1
            retry_attempted = True
            reason = errors if errors else "zero_vector"
            print(f"      [retry {retry_count}/{args.retry_on_invalid}] reason={reason}")
            retry_messages = _apply_prefill(base_messages, use_prefill=True)
            result = call_chat_with_retries(
                base_url=args.ollama_base_url,
                api_key=api_key,
                timeout_seconds=args.timeout_seconds,
                messages=retry_messages,
                model=args.model,
                temperature=base_temp,
                top_p=args.top_p,
                max_completion_tokens=args.max_tokens,
                rate_limiter=rate_limiter,
                retry_per_prompt=args.retry_per_prompt,
                rate_limit_wait_base_seconds=args.rate_limit_wait_base_seconds,
                rate_limit_wait_max_seconds=args.rate_limit_wait_max_seconds,
                hard_cooldown_seconds=args.hard_cooldown_seconds,
                request_logprobs=False,
                response_schema=None,
                context_size=args.context_size,
            )
            raw = _reconstruct_prefilled(result.content, use_prefill=True)
            parsed, errors, vector = parse_multilabel_response(raw, categories)

            v_sum = sum(vector.values())
            if v_sum > best_vector_sum:
                best_vector_sum = v_sum
                best_vector = vector
                best_parsed = parsed
            if parsed is not None:
                any_attempt_parsed = True

        # Use the best vector seen, not just the last attempt's.
        vector = best_vector
        parsed = best_parsed

        # Maintenance-reason axis: read from the same best parsed payload. This
        # never drives retries (categories do); missing block -> zero vector.
        maint_vector = maintenance_vector_from_payload(parsed, maintenance_types)

        chunk_duration = time.time() - chunk_start
        is_zero = not any(v > 0.0 for v in vector.values())
        # If the model returned an all-zero vector on a chunk that genuinely changed lines, it likely ignored the instructions. 
        if is_zero and chunk_changed > 0:
            status = "fallback"
        # if is_zero and chunk_changed > 0 and retry_attempted:
        elif not errors:
            status = "ok"
        # Even when errors are present, the model may have returned a valid vector with some signal (e.g. some categories scored above the flagging threshold) that we can still use, so we consider that a "recovered" outcome rather than a failure.
        elif _has_usable_signal(vector):
            status = "recovered"
        # When retries were attempted but the model never returned valid JSON or a usable signal, we consider that a "retried_failed" outcome: the retry logic worked as intended to surface a degraded result that we can still use for aggregation, but the final outcome is still a failure since the model did not follow instructions even after retries.
        elif retry_attempted:
            status = "retried_failed"
        # When the model returned errors and no usable signal, but we did not attempt a retry (either because retries are disabled or because the retry condition is only triggered on certain errors that were not present), we consider that a "fallback" outcome since we will still use the all-zero vector for aggregation but it indicates the model did not follow instructions and we did not even attempt to nudge it with a retry.
        else:
            status = "fallback"
        print(
            f"      chunk {chunk.index + 1}/{chunk.total} | status={status} "
            f"| changed_lines={chunk_changed} | errors={errors or '-'} | elapsed={chunk_duration:.1f}s"
        )
        
        if errors and "invalid_json" in errors:
            flat = raw.replace("\n", " ") if raw else ""
            if not flat:
                print("      [invalid-json debug] <empty response>")
            else:
                print(f"      [invalid-json debug len={len(flat)}] tail={flat[-180:]}")
        if is_zero and chunk_changed > 0:
            changed_only = " | ".join(
                line.rstrip()
                for line in chunk.text.splitlines()
                if (line.startswith("+") or line.startswith("-"))
                and not line.startswith("+++") and not line.startswith("---")
            )
            print(f"      [zero-vector debug] changed={changed_only[:400]}")
            preview = raw.replace("\n", " ")[:300] if raw else "<empty>"
            print(f"      [zero-vector debug] raw={preview}")

        chunk_records.append(
            {
                "index": chunk.index,
                "total": chunk.total,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "raw": raw,
                "parsed": parsed,
                "validation_errors": errors,
                "status": status,
                "retry_attempted": retry_attempted,
                "vector": vector,
                "maint_vector": maint_vector,
                "any_attempt_parsed": any_attempt_parsed,
            }
        )
        chunk_vectors.append(vector)
        chunk_maint_vectors.append(maint_vector)

    final_vector = aggregate_chunk_vectors(
        chunk_vectors,
        categories,
        aggregation=args.chunk_aggregation,
    )

    # Aggregate the maintenance-reason axis the same way as categories, then
    # derive the primary leaf type and its parent class/base type in code.
    final_maint_vector = aggregate_chunk_vectors(
        chunk_maint_vectors,
        maintenance_types,
        aggregation=args.chunk_aggregation,
    )


    all_chunks_failed = not any(cr.get("any_attempt_parsed", False) for cr in chunk_records)
    has_changed_lines = any(
        (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
        for line in diff_text.splitlines()
    )
    fallback_applied = False
    if not any(v > 0.0 for v in final_vector.values()) and (all_chunks_failed or has_changed_lines):
        _changed_lc = "\n".join(
            line[1:]
            for line in diff_text.splitlines()
            if (line.startswith("+") or line.startswith("-"))
            and not line.startswith("+++") and not line.startswith("---")
        ).lower()

        _fallback_rules = [
            (r"(?:^|[\s`])/[a-z][\w-]+|slash command|custom command|\bmcp\b|\bagent\b|claude|copilot|\bllm\b", "AI Integration"),
            (r"pull request|\bpr\b|merge queue|\bcommit\b|\bbranch\b|code review", "Development Process"),
            (r"\btest|pytest|jest|vitest|coverage", "Testing"),
            (r"ci/cd|\bci\b|workflow|deploy|release|pipeline", "DevOps"),
            (r"npm run|cargo build|\bmake\b|compile|build command", "Build and Run"),
            (r"\.env\b|environment variable|config file|\bsetup\b|\binstall\b", "Conf.&Env."),
            (r"security|vulnerab|\bauth\b|token|secret", "Security"),
            (r"performance|latency|cache|optimi", "Performance"),
            (r"architecture|module|package|layer|design pattern", "Architecture"),
            (r"readme|changelog|\bdocs?\b|documentation|https?://", "Documentation"),
        ]
        fallback_cat = "Impl. Details"
        for _pat, _cat in _fallback_rules:
            if _cat in categories and re.search(_pat, _changed_lc):
                fallback_cat = _cat
                break
        if fallback_cat not in categories:
            fallback_cat = "Impl. Details" if "Impl. Details" in categories else (
                categories[0] if categories else None
            )
        if fallback_cat is not None and fallback_cat in categories:
            # Assign a minimal rescue score so the diff isn't left unclassified.
            final_vector = {c: 0.0 for c in categories}
            final_vector[fallback_cat] = 0.20
            fallback_applied = True
            print(f"    [signals-fallback] zero vector rescued with {fallback_cat} default")


    if fallback_applied:
        diff_status = "fallback"
    else:
        _status_severity = {"ok": 0, "recovered": 1, "retried_failed": 2, "fallback": 3}
        diff_status = max(
            (cr.get("status", "ok") for cr in chunk_records),
            key=lambda s: _status_severity.get(s, 0),
            default="ok",
        )


    if any(v > 0.0 for v in final_vector.values()):
        primary_category = max(final_vector.items(), key=lambda kv: kv[1])[0]
    else:
        primary_category = "System Overview"

    rationale_map = _extract_rationale_map(chunk_records, categories)
    flagged = [
        {"category": c, "score": s, "rationale": rationale_map.get(c, "")}
        for c, s in sorted(final_vector.items(), key=lambda kv: kv[1], reverse=True)
        if s >= args.per_category_threshold
    ]

    if not flagged and any(v > 0.0 for v in final_vector.values()):
        top_category, top_score = max(final_vector.items(), key=lambda kv: kv[1])
        flagged = [
            {
                "category": top_category,
                "score": top_score,
                "rationale": rationale_map.get(top_category, ""),
            }
        ]


    _added_text = "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ).lower()
    _removed_text = "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ).lower()

    def _cat_prefix(c: str) -> str:
        for _p in _CATEGORY_PATTERNS.get(c, []):
            if re.search(_p, _added_text):
                return "+"
        for _p in _CATEGORY_PATTERNS.get(c, []):
            if re.search(_p, _removed_text):
                return "-"
        # No keyword match: fall back to diff direction
        return "+" if _added_text else "-"

    non_zero = [(c, s) for c, s in final_vector.items() if s > 0.0]
    non_zero.sort(key=lambda kv: kv[1], reverse=True)
    categories_line = ", ".join(f"{_cat_prefix(c)}{c}={s:.2f}" for c, s in non_zero) or "-"
    flagged_line = ", ".join(f"{f['category']}={f['score']:.2f}" for f in flagged) or "-"
    # Self-contained per-diff line: include the diff identity so it stays
    # readable when run_multi_model.py interleaves several models' output
    # (the wrapper already prefixes every line with the model name).
    print(
        f"    {filename} | {diff_id} | status={diff_status} | "
        f"primary={primary_category} | flagged({len(flagged)})={flagged_line} | "
        f"categories({len(non_zero)})={categories_line}"
    )

    # Derive the maintenance-reason result: primary leaf = argmax of the
    # aggregated vector; parent class and base type follow deterministically
    # from the taxonomy (the adaptive split makes the parent unambiguous).
    maint_rationale_map = _extract_rationale_map(
        chunk_records, maintenance_types, key="maintenance_types"
    )
    if any(v > 0.0 for v in final_maint_vector.values()):
        primary_maintenance_type = max(final_maint_vector.items(), key=lambda kv: kv[1])[0]
    else:
        primary_maintenance_type = None
    maintenance_class = leaf_to_class.get(primary_maintenance_type or "", "")
    maintenance_base_type = leaf_to_base.get(
        primary_maintenance_type or "", primary_maintenance_type or ""
    )
    maintenance_block = {
        "primary_maintenance_type": primary_maintenance_type,
        "maintenance_class": maintenance_class,
        "maintenance_base_type": maintenance_base_type,
        "type_scores": final_maint_vector,
        "rationale": maint_rationale_map.get(primary_maintenance_type or "", ""),
    }

    # Display base type + class to avoid the redundant "(enhancement) (Enhancement)"
    # doubling: the leaf name already encodes the class, so show the base type.
    maint_line = (
        f"{maintenance_base_type} ({maintenance_class})"
        if primary_maintenance_type
        else "-"
    )
    print(f"    maintenance | primary={maint_line}")

    return {
        "diff_id": diff_id,
        "commit_hash": commit_hash,
        "commit_message": commit_message,
        "timestamp": diff.get("timestamp", ""),
        "filename": filename,
        "status": diff_status,
        # "signals": {"added": added_signals, "removed": removed_signals},
        "primary": {
            "model": args.model,
            "primary_category": primary_category,
            "summary": _best_summary(chunk_records, primary_category),
            "validation_errors": _aggregate_validation_errors(chunk_records),
        },
        "category_scores": final_vector,
        "flagged_categories": flagged,
        "maintenance": maintenance_block,
        "chunk_plan": plan.to_metadata(),
        "chunk_records": chunk_records if args.include_chunk_details else [],
    }



# Minimum tokens reserved for diff content + model response.
_MIN_DIFF_RESPONSE_TOKENS = 900


def _check_context_budget(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    context_size: int | None,
) -> int | None:
    """Print a startup context report and warn when headroom is insufficient.

    Returns the effective context size (tokens) so the caller can derive a
    safe chunk size. Returns None when the context cannot be determined.

    ``context_size`` is the user's explicit override (None = use model default).
    """
    system_tokens = len(system_prompt) // 4
    model_ctx = fetch_model_context_size(base_url, api_key, model)

    if context_size is None:
        effective_ctx = model_ctx
    elif model_ctx is not None:
        effective_ctx = min(context_size, model_ctx)
    else:
        effective_ctx = context_size

    budget = (effective_ctx - system_tokens) if effective_ctx else None
    ctx_source = f"{model_ctx} (model)" if model_ctx else "unknown"
    override_note = f"--context-size={context_size}" if context_size else "no override"

    print(
        f"[context] system_prompt={system_tokens} tok | "
        f"{override_note} | model num_ctx={ctx_source} | "
        f"effective={effective_ctx or 'unknown'} | "
        f"budget={budget if budget is not None else 'unknown'} tok"
    )

    if budget is not None and budget < _MIN_DIFF_RESPONSE_TOKENS:
        print(
            f"  WARNING: only {budget} token available for diff+response "
            f"(minimum recommended: {_MIN_DIFF_RESPONSE_TOKENS}).\n"
            f"  Expect 'invalid_json' and missing categories.\n"
            f"  Fix — create a model with larger context:\n"
            f"    echo 'FROM {model}\\nPARAMETER num_ctx 8192' | "
            f"ollama create {model}-ctx8k -f -\n"
            f"  Then re-run with --model {model}-ctx8k --context-size 8192",
            file=sys.stderr,
        )
    elif budget is not None and budget < 1500:
        print(
            f"  NOTE: {budget} token budget is sufficient only for small diffs. "
            f"Large diffs may still produce missing-category errors.",
        )

    return effective_ctx




MISSING_CATEGORIES_ERROR = "categories_missing_or_not_list"


def _find_latest_primary_file(model_dir: Path) -> Path | None:
    """Return the most recent primary_*.jsonl in model_dir, or None.

    Files are named primary_YYYYMMDD_HHMMSS.jsonl so lexicographic order
    equals chronological order (fixed-width, most-significant parts first).
    """
    if not model_dir.exists():
        return None
    candidates = sorted(
        model_dir.glob("primary_*.jsonl"),
        key=lambda p: p.stem,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _record_has_missing_categories_error(record: dict) -> bool:
    """Return True if any chunk's validation errors include the missing-list code.

    The record may carry per-chunk validation errors either at the top
    level (``record["primary"]["validation_errors"]`` when there is a
    single chunk) or inside ``record["chunk_records"]``. We check both.

    Top-level errors are emitted by :func:`_aggregate_validation_errors`
    and are tagged as ``"chunk<i>:<code>"``; chunk-level errors are
    plain code strings. Both forms are accepted.
    """
    if not record:
        return False

    def _matches(err: str) -> bool:
        if not isinstance(err, str):
            return False
        if err == MISSING_CATEGORIES_ERROR:
            return True
        # Tagged form: "chunk0:categories_missing_or_not_list"
        if err.startswith("chunk") and err.endswith(MISSING_CATEGORIES_ERROR):
            return True
        return False

    primary = record.get("primary") or {}
    if isinstance(primary, dict):
        for err in primary.get("validation_errors") or []:
            if _matches(err):
                return True
    for chunk in record.get("chunk_records") or []:
        if not isinstance(chunk, dict):
            continue
        for err in chunk.get("validation_errors") or []:
            if _matches(err):
                return True
    return False




def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_path = args.env_file
    if not env_path.is_absolute() and not env_path.exists():
        candidate = repo_root / env_path
        if candidate.exists():
            env_path = candidate

    label_path = args.label_descriptions
    if not label_path.is_absolute() and not label_path.exists():
        candidate = repo_root / label_path
        if candidate.exists():
            label_path = candidate

    mr_path = args.maintenance_descriptions
    if not mr_path.is_absolute() and not mr_path.exists():
        candidate = repo_root / mr_path
        if candidate.exists():
            mr_path = candidate

    api_key = load_ollama_api_key(env_path)
    if not api_key:
        if not env_path.exists():
            raise RuntimeError(
                f"Env file not found: {env_path} (resolved from --env-file="
                f"{args.env_file}). Check the path — e.g. '.contextWare\\.env' "
                f"is a file inside the .contextWare folder, not '.contextWare.env'."
            )
        raise RuntimeError(
            f"OLLAMA_API_KEY is missing or empty in {env_path}. "
            f"Add a line 'OLLAMA_API_KEY=<your-key>'."
        )

    label_categories, label_descriptions, label_examples = load_label_descriptions(label_path)
    categories = load_categories(args.categories, label_categories or DEFAULT_CATEGORIES)
    if not categories:
        categories = list(DEFAULT_CATEGORIES)

    # Maintenance-reason taxonomy (the "why" axis). Falls back to the built-in
    # defaults in acf_prompt when modification_request.json is unavailable.
    maint = load_maintenance_descriptions(mr_path)
    maintenance_types = maint["types"] or list(DEFAULT_MAINTENANCE_TYPES)
    leaf_to_class = maint["leaf_to_class"] or dict(DEFAULT_LEAF_TO_CLASS)
    leaf_to_base = maint["leaf_to_base"] or dict(DEFAULT_LEAF_TO_BASE)

    system_prompt = get_system_prompt(
        categories,
        label_descriptions,
        label_examples,
        maintenance_types=maintenance_types,
        maintenance_descriptions=maint["descriptions"],
        maintenance_examples=maint["examples"],
        maintenance_classes=maint["classes"] or None,
        maintenance_class_descriptions=maint["class_descriptions"],
    )
    effective_ctx = _check_context_budget(
        base_url=args.ollama_base_url,
        api_key=api_key,
        model=args.model,
        system_prompt=system_prompt,
        context_size=args.context_size,
    )


    if effective_ctx is not None:
        system_tokens = len(system_prompt) // 4
        _RESPONSE_BUDGET = args.max_tokens
        _OVERHEAD = 200
        safe_diff_tokens = max(500, effective_ctx - system_tokens - _RESPONSE_BUDGET - _OVERHEAD)
        safe_diff_chars = safe_diff_tokens * 4
        if args.max_diff_chars > safe_diff_chars:
            print(
                f"[chunker] --max-diff-chars={args.max_diff_chars} exceeds safe limit "
                f"({safe_diff_chars} chars) for effective_ctx={effective_ctx} tok "
                f"→ reduced to {safe_diff_chars}"
            )
            args.max_diff_chars = safe_diff_chars
        else:
            print(
                f"[chunker] max_diff_chars={args.max_diff_chars} | "
                f"safe_limit={safe_diff_chars} | chunk_size=ok"
            )

    input_paths = resolve_input_paths(args.diffs_json)
    if args.repo_filter:
        input_paths = [p for p in input_paths if args.repo_filter in p.stem]
    if not input_paths:
        print(f"No input files matched --repo-filter={args.repo_filter!r}", file=sys.stderr)
        sys.exit(1)

    timestamp_str = get_run_timestamp(args.timestamp)
    print(f"[output] writing to {args.output_dir.resolve()}")
    rate_limiter = SlidingRateLimiter(
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        min_interval_seconds=args.min_request_interval_seconds,
    )

    if args.print_prompt:
        # Show a representative prompt using the first repo / first diff.
        for diffs_path in input_paths:
            repo_label, diffs = load_repo_diffs(diffs_path)
            if not diffs:
                continue
            sample = diffs[0]
            # _added_sigs, _removed_sigs = build_signals(...)
            # signals = list(dict.fromkeys(_added_sigs + _removed_sigs))
            user_prompt = build_user_prompt(
                chunk_text=str(sample.get("diff_text", ""))[: args.max_diff_chars],
                filename=str(sample.get("filename", "")),
                commit_message=str(sample.get("commit_message", "")),
                commit_date=str(sample.get("timestamp", "")),
                chunk_index=0,
                chunk_total=1,
                char_start=0,
                char_end=min(args.max_diff_chars, len(str(sample.get("diff_text", "")))),
                # signals=signals,
            )
            print("\n--- SYSTEM PROMPT ---\n")
            print(system_prompt)
            print("\n--- USER PROMPT ---\n")
            print(user_prompt)
            print("\n--- END PROMPT ---\n")
            break

    for repo_index, diffs_path in enumerate(input_paths, start=1):
        repo_label, diffs = load_repo_diffs(diffs_path)
        if not repo_label:
            repo_label = diffs_path.stem
        if args.limit is not None:
            diffs = diffs[: args.limit]

        primary_output_path, eval_output_path = build_output_paths(
            args.output_dir,
            args.model,
            timestamp_str,
        )
        if args.resume:
            latest = _find_latest_primary_file(primary_output_path.parent)
            processed_ids = load_processed_ids(latest) if latest else set()
            if latest:
                print(f"  [resume] {latest.name} — {len(processed_ids)} id già processati")
            else:
                print("  [resume] nessun file precedente trovato, si parte da zero")
        else:
            processed_ids = set()


        repo_fn_totals: dict[str, dict[str, int]] = {}
        for d in diffs:
            r = str(d.get("repo", ""))
            fn = str(d.get("filename", "unknown"))
            repo_fn_totals.setdefault(r, {}).setdefault(fn, 0)
            repo_fn_totals[r][fn] += 1

        print(f"[repo {repo_index}/{len(input_paths)}] {repo_label}  ({len(diffs)} diffs)")

        with (
            primary_output_path.open("a", encoding="utf-8") as primary_handle,
            eval_output_path.open("a", encoding="utf-8") as eval_handle,
        ):
            repo_fn_counters: dict[str, dict[str, int]] = {}
            for index, diff in enumerate(diffs, start=1):
                diff_id = str(diff.get("diff_id", f"diff-{index}"))
                if diff_id in processed_ids:
                    continue

                repo_name = str(diff.get("repo", ""))
                filename = str(diff.get("filename", "unknown"))
                repo_fn_counters.setdefault(repo_name, {}).setdefault(filename, 0)
                repo_fn_counters[repo_name][filename] += 1
                file_idx = repo_fn_counters[repo_name][filename]
                file_total = repo_fn_totals.get(repo_name, {}).get(filename, "?")
                file_progress = f"{filename} {file_idx}/{file_total}"
                repo_prefix = f"{repo_name} | " if repo_name else ""
                print(f"  [{index}/{len(diffs)}] {repo_prefix}{file_progress} | {diff_id}")
                zero_flags_max = args.retry_on_invalid if args.retry_on_zero_flags else 0
                max_diff_retries = max(
                    0,
                    int(args.diff_retry_on_missing_list),
                    zero_flags_max,
                )
                attempts_allowed = 1 + max_diff_retries  # initial + retries
                record: dict | None = None
                diff_retry_count = 0

                for attempt in range(1, attempts_allowed + 1):
                    try:
                        record = classify_diff(
                            diff=diff,
                            args=args,
                            categories=categories,
                            maintenance_types=maintenance_types,
                            leaf_to_class=leaf_to_class,
                            leaf_to_base=leaf_to_base,
                            system_prompt=system_prompt,
                            api_key=api_key,
                            rate_limiter=rate_limiter,
                        )
                    except Exception as exc: 
                        print(f"    ERROR classifying {diff_id}: {exc}", file=sys.stderr)
                        record = {
                            "diff_id": diff_id,
                            "commit_hash": str(diff.get("commit_hash", "")),
                            "commit_message": str(diff.get("commit_message", "")),
                            "timestamp": diff.get("timestamp", ""),
                            "filename": str(diff.get("filename", "")),
                            "primary": {
                                "model": args.model,
                                "primary_category": "System Overview",
                                "summary": "",
                                "validation_errors": ["pipeline_exception"],
                                "error": str(exc),
                            },
                            "category_scores": {c: 0.0 for c in categories},
                            "flagged_categories": [],
                            "maintenance": {
                                "primary_maintenance_type": None,
                                "maintenance_class": "",
                                "maintenance_base_type": "",
                                "type_scores": {t: 0.0 for t in maintenance_types},
                                "rationale": "",
                            },
                            "chunk_plan": {"truncated": False, "original_chars": 0, "chunk_count": 0},
                            "chunk_records": [],
                        }

                    _has_missing = _record_has_missing_categories_error(record)
                    _has_zero_flags = (
                        args.retry_on_zero_flags
                        and record is not None
                        and len(record.get("flagged_categories", [])) == 0
                        and not any(
                            v > 0.0
                            for v in record.get("category_scores", {}).values()
                        )
                    )

                    should_retry = attempt < attempts_allowed and (_has_missing or _has_zero_flags)
                    if not should_retry:
                        break

                    diff_retry_count += 1
                    backoff = min(30, 2 ** (diff_retry_count - 1))
                    reason = "categories_missing_or_not_list" if _has_missing else "zero_flags"
                    print(
                        f"    retry {diff_retry_count}/{max_diff_retries} "
                        f"for {diff_id} in {backoff}s ({reason})"
                    )
                    time.sleep(backoff)

                if record is None:  # pragma: no cover - defensive
                    continue

                if _record_has_missing_categories_error(record):
                    print(
                        f"    giving up on {diff_id} after {diff_retry_count} "
                        f"diff-retry attempt(s); writing zero vector"
                    )
                    record.setdefault("primary", {})["diff_retry_count"] = diff_retry_count
                    record["primary"]["diff_retry_exhausted"] = True
                else:
                    record.setdefault("primary", {})["diff_retry_count"] = diff_retry_count

                if args.include_patch:
                    record["diff_text"] = truncate_text(
                        str(diff.get("diff_text", "")),
                        args.max_diff_chars,
                    )

                # Eval payload is a flat summary for downstream evaluation.
                _maint = record.get("maintenance", {}) or {}
                eval_record = {
                    "diff_id": diff_id,
                    "primary_category": record.get("primary", {}).get("primary_category"),
                    "category_scores": record.get("category_scores", {}),
                    "flagged_categories": record.get("flagged_categories", []),
                    "primary_maintenance_type": _maint.get("primary_maintenance_type"),
                    "maintenance_class": _maint.get("maintenance_class", ""),
                    "maintenance_base_type": _maint.get("maintenance_base_type", ""),
                    "maintenance_type_scores": _maint.get("type_scores", {}),
                    "chunk_count": record.get("chunk_plan", {}).get("chunk_count", 0),
                }

                primary_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                primary_handle.flush()
                eval_handle.write(json.dumps(eval_record, ensure_ascii=True) + "\n")
                eval_handle.flush()


if __name__ == "__main__":
    main()
