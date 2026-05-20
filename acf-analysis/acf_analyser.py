from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from acf_io import (
    build_repo_output_paths,
    get_run_timestamp,
    load_categories,
    load_label_descriptions,
    load_ollama_api_key,
    load_processed_ids,
    load_repo_diffs,
    truncate_text,
)
from acf_parsing import (
    build_category_confidences,
    build_fallback_primary,
    parse_primary_response,
)
from acf_prompt import DEFAULT_MODEL, build_primary_prompt, build_signals, select_category_sample
from ollama_client import SlidingRateLimiter, call_chat_with_retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the primary ACF prompt to extract rules and key phrases from diffs."
    )
    parser.add_argument(
        "--diffs-json",
        type=Path,
        default=Path("acf_diffs.json"),
        help="Path to diffs JSON file or a directory of JSON files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("acf-outputs"))
    parser.add_argument(
        "--label-descriptions",
        type=Path,
        default=Path("categories.json"),
        help="Path to categories.json with category names and descriptions.",
    )
    parser.add_argument(
        "--category-sample-size",
        type=int,
        default=6,
        help="Number of category candidates to include in the prompt (0 = all).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Override output timestamp (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the Ollama API",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--max-diff-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--retry-per-prompt", type=int, default=3)
    parser.add_argument("--requests-per-minute", type=int, default=24)
    parser.add_argument("--tokens-per-minute", type=int, default=12000)
    parser.add_argument("--min-request-interval-seconds", type=float, default=2.0)
    parser.add_argument("--rate-limit-wait-base-seconds", type=int, default=12)
    parser.add_argument("--rate-limit-wait-max-seconds", type=int, default=45)
    parser.add_argument("--hard-cooldown-seconds", type=int, default=90)
    parser.add_argument(
        "--retry-on-invalid",
        type=int,
        default=1,
        help="Retry the same prompt when JSON or schema validation fails.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip diffs already in output")
    parser.add_argument("--include-patch", action="store_true", help="Include diff text in output records")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the fully rendered prompt for each diff.",
    )
    return parser.parse_args()


def resolve_input_paths(diffs_path: Path) -> list[Path]:
    if not diffs_path.exists():
        raise FileNotFoundError(f"Diffs input not found: {diffs_path}")
    if diffs_path.is_dir():
        candidates = sorted(path for path in diffs_path.glob("*.json") if path.is_file())
        if not candidates:
            raise ValueError(f"No JSON files found in {diffs_path}")
        return candidates
    return [diffs_path]


def main() -> None:
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

    api_key = load_ollama_api_key(env_path)
    if not api_key:
        raise RuntimeError(f"Missing OLLAMA_API_KEY in {env_path}")

    label_categories, label_descriptions = load_label_descriptions(label_path)
    categories = load_categories(args.categories, label_categories)
    category_descriptions = {
        name: desc for name, desc in label_descriptions.items() if name in categories
    }
    fallback_category = categories[0] if categories else "System Overview"

    input_paths = resolve_input_paths(args.diffs_json)

    timestamp_str = get_run_timestamp(args.timestamp)

    rate_limiter = SlidingRateLimiter(
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        min_interval_seconds=args.min_request_interval_seconds,
    )

    for repo_index, diffs_path in enumerate(input_paths, start=1):
        repo_label, diffs = load_repo_diffs(diffs_path)
        if not repo_label:
            repo_label = diffs_path.stem
        if args.limit is not None:
            diffs = diffs[: args.limit]

        primary_output_path, eval_output_path = build_repo_output_paths(
            args.output_dir,
            args.model,
            timestamp_str,
            repo_label,
            repo_index,
        )
        processed_ids = load_processed_ids(primary_output_path) if args.resume else set()

        print(f"[repo {repo_index}/{len(input_paths)}] {repo_label}")

        with primary_output_path.open("a", encoding="utf-8") as primary_handle, eval_output_path.open(
            "a", encoding="utf-8"
        ) as eval_handle:
            for index, diff in enumerate(diffs, start=1):
                diff_id = str(diff.get("diff_id", f"diff-{index}"))
                if diff_id in processed_ids:
                    continue

                diff_text = truncate_text(str(diff.get("diff_text", "")), args.max_diff_chars)
                commit_hash = str(diff.get("commit_hash", ""))
                commit_message = str(diff.get("commit_message", ""))
                filename = str(diff.get("filename", ""))
                signals = build_signals(diff_text, filename, commit_message)
                categories_for_prompt = select_category_sample(
                    categories,
                    signals,
                    args.category_sample_size,
                )

                print(f"[{index}/{len(diffs)}] Processing {diff_id}")

                primary_prompt = build_primary_prompt(
                    diff_text=diff_text,
                    commit_message=commit_message,
                    filename=filename,
                    categories=categories_for_prompt,
                    category_descriptions=category_descriptions,
                    signals=signals,
                )
                if args.print_prompt:
                    print("\n--- PROMPT START ---\n")
                    print(primary_prompt)
                    print("\n--- PROMPT END ---\n")

                primary_result = call_chat_with_retries(
                    base_url=args.ollama_base_url,
                    api_key=api_key,
                    timeout_seconds=args.timeout_seconds,
                    prompt=primary_prompt,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_completion_tokens=args.max_tokens,
                    rate_limiter=rate_limiter,
                    retry_per_prompt=args.retry_per_prompt,
                    rate_limit_wait_base_seconds=args.rate_limit_wait_base_seconds,
                    rate_limit_wait_max_seconds=args.rate_limit_wait_max_seconds,
                    hard_cooldown_seconds=args.hard_cooldown_seconds,
                    request_logprobs=False,
                )
                primary_raw = primary_result.content
                raw_initial = primary_raw
                retry_attempted = False

                primary_parsed, validation_errors = parse_primary_response(
                    primary_raw,
                    categories=categories,
                    fallback_category=fallback_category,
                    threshold=0.50,
                    confidence_categories=categories_for_prompt,
                )

                if primary_parsed is None or validation_errors:
                    if args.retry_on_invalid > 0:
                        retry_attempted = True
                        primary_result = call_chat_with_retries(
                            base_url=args.ollama_base_url,
                            api_key=api_key,
                            timeout_seconds=args.timeout_seconds,
                            prompt=primary_prompt,
                            model=args.model,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_completion_tokens=args.max_tokens,
                            rate_limiter=rate_limiter,
                            retry_per_prompt=args.retry_per_prompt,
                            rate_limit_wait_base_seconds=args.rate_limit_wait_base_seconds,
                            rate_limit_wait_max_seconds=args.rate_limit_wait_max_seconds,
                            hard_cooldown_seconds=args.hard_cooldown_seconds,
                            request_logprobs=False,
                        )
                        primary_raw = primary_result.content
                        primary_parsed, validation_errors = parse_primary_response(
                            primary_raw,
                            categories=categories,
                            fallback_category=fallback_category,
                            threshold=0.50,
                            confidence_categories=categories_for_prompt,
                        )

                if primary_parsed is None or validation_errors:
                    primary_parsed = build_fallback_primary(primary_raw, fallback_category)

                is_valid = not validation_errors
                if retry_attempted:
                    parse_status = "retried_ok" if is_valid else "fallback"
                else:
                    parse_status = "ok" if is_valid else "fallback"

                key_phrases = primary_parsed.get("key_phrases") if isinstance(primary_parsed, dict) else []
                rules = primary_parsed.get("rules") if isinstance(primary_parsed, dict) else []
                category_confidences = build_category_confidences(
                    categories_for_prompt,
                    primary_parsed,
                    signals,
                )

                eval_payload = {
                    "key_phrases": key_phrases if isinstance(key_phrases, list) else [],
                    "rules": rules if isinstance(rules, list) else [],
                    "categories": categories_for_prompt,
                }

                primary_record = {
                    "diff_id": diff_id,
                    "commit_hash": commit_hash,
                    "commit_message": commit_message,
                    "timestamp": diff.get("timestamp", ""),
                    "filename": filename,
                    "signals": signals,
                    "primary": {
                        "model": args.model,
                        "raw": primary_raw,
                        "parsed": primary_parsed,
                        "parse_status": parse_status,
                        "validation_errors": validation_errors,
                        "retry_attempted": retry_attempted,
                    },
                    "category_confidences": category_confidences,
                }
                if retry_attempted:
                    primary_record["primary"]["raw_initial"] = raw_initial
                if args.include_patch:
                    primary_record["diff_text"] = diff_text

                eval_record = {
                    "diff_id": diff_id,
                    "signals": signals,
                    "eval_payload": eval_payload,
                }

                primary_handle.write(json.dumps(primary_record, ensure_ascii=True) + "\n")
                primary_handle.flush()
                eval_handle.write(json.dumps(eval_record, ensure_ascii=True) + "\n")
                eval_handle.flush()


if __name__ == "__main__":
    main()