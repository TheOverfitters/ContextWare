"""Run the ACF classifier pipeline over several LLMs at once.

This is a thin orchestrator around ``acf_analyser.py``: it launches one
classification run per model (each in its own subprocess) so the three
cloud LLMs used for the ambiguity study are evaluated on the *same* diffs
with the *same* run timestamp. Because every run writes to its own
``<output-dir>/<model>/primary_<timestamp>.jsonl``, the outputs never
collide and can later be aligned by ``diff_id`` for the agreement analysis.

The pipeline code itself is reused unchanged: any argument this wrapper
does not recognise is forwarded verbatim to ``acf_analyser.py``. Only
``--model`` and ``--timestamp`` are injected by the wrapper (one model per
run, one shared timestamp for all runs), so passing them yourself is an
error.

Examples
--------
Run the three ambiguity-study models concurrently at temperature 0::

    python run_multi_model.py --diffs-json ../outputs/git_history_diffs.jsonl

Limit to a quick smoke test of 5 diffs per repo::

    python run_multi_model.py --limit 5

Forward extra analyser flags (anything unknown is passed through)::

    python run_multi_model.py --resume --per-category-threshold 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Reuse the analyser's own helpers so output paths/timestamps match exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acf_io import build_output_paths, get_run_timestamp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSER = SCRIPT_DIR / "acf_analyser.py"

# The three cloud LLMs used for the ambiguity study (PDF section 4).
DEFAULT_MODELS = [
    "kimi-k2.7-code:cloud",
    "qwen3.5:cloud",
    "glm-5.2:cloud",
]

# Default input + output mirror the convention already used in this repo:
# diffs come from outputs/git_history_diffs.jsonl and per-model results land
# under acf-analysis/acf-outputs/<model>/.
DEFAULT_DIFFS = SCRIPT_DIR.parent / "outputs" / "git_history_diffs.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "acf-outputs"

# Serialise console writes so interleaved subprocess lines stay readable.
_PRINT_LOCK = threading.Lock()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run acf_analyser.py over several models at once (one subprocess "
            "per model, shared timestamp). Unknown args are forwarded to the "
            "analyser."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to run (default: the three ambiguity-study cloud LLMs).",
    )
    parser.add_argument(
        "--diffs-json",
        type=Path,
        default=DEFAULT_DIFFS,
        help=f"Diffs input forwarded to every run (default: {DEFAULT_DIFFS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root forwarded to every run (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature forwarded to every run. Default 0.0 to "
            "isolate inter-model divergence from within-model sampling noise."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help=(
            "Shared run timestamp (YYYYMMDD_HHMMSS) used by every model so "
            "their output files line up. Default: generated once at startup."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max models to run concurrently (default: all of them).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run models one at a time instead of concurrently.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run for each model and exit.",
    )
    args, passthrough = parser.parse_known_args()

    # Guard against arguments the wrapper owns being passed twice.
    for owned in ("--model", "--timestamp", "--diffs-json", "--temperature", "--output-dir"):
        if owned in passthrough:
            parser.error(
                f"{owned} is managed by the wrapper; pass it via the wrapper's "
                f"own option, not as a forwarded analyser argument."
            )
    return args, passthrough


def build_command(
    model: str,
    timestamp: str,
    args: argparse.Namespace,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        # -u: unbuffered stdout so the analyser's per-diff print() lines stream
        # to our pipe in real time instead of arriving in delayed block-buffered
        # bursts (logging already flushes per-message; plain print() does not).
        "-u",
        str(ANALYSER),
        "--model", model,
        "--timestamp", timestamp,
        "--diffs-json", str(args.diffs_json),
        "--output-dir", str(args.output_dir),
        "--temperature", str(args.temperature),
    ]
    cmd.extend(passthrough)
    return cmd


def run_one(
    model: str,
    timestamp: str,
    args: argparse.Namespace,
    passthrough: list[str],
) -> tuple[str, int]:
    """Run the analyser for a single model, streaming its output with a prefix."""
    cmd = build_command(model, timestamp, args, passthrough)
    prefix = f"[{model}]"

    with _PRINT_LOCK:
        print(f"{prefix} START -> {' '.join(cmd)}", flush=True)

    # cwd = SCRIPT_DIR so acf_analyser's sibling imports (acf_chunker, ...)
    # resolve and its repo-root .env lookup works.
    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        with _PRINT_LOCK:
            print(f"{prefix} {line.rstrip()}", flush=True)
    returncode = proc.wait()

    with _PRINT_LOCK:
        status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
        print(f"{prefix} DONE -> {status}", flush=True)
    return model, returncode


def main() -> None:
    args, passthrough = parse_args()

    if not ANALYSER.exists():
        sys.exit(f"Cannot find analyser at {ANALYSER}")
    if not args.diffs_json.exists():
        sys.exit(f"Diffs input not found: {args.diffs_json}")

    timestamp = get_run_timestamp(args.timestamp)

    print(f"[multi] models      : {', '.join(args.models)}")
    print(f"[multi] diffs-json  : {args.diffs_json}")
    print(f"[multi] output-dir  : {args.output_dir}")
    print(f"[multi] timestamp   : {timestamp}")
    print(f"[multi] temperature : {args.temperature}")
    print(f"[multi] mode        : {'sequential' if args.sequential else 'concurrent'}")

    if args.dry_run:
        for model in args.models:
            cmd = build_command(model, timestamp, args, passthrough)
            print(f"[dry-run] {model}: {' '.join(cmd)}")
        return

    results: list[tuple[str, int]] = []
    if args.sequential:
        for model in args.models:
            results.append(run_one(model, timestamp, args, passthrough))
    else:
        max_workers = args.max_workers or len(args.models)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(run_one, model, timestamp, args, passthrough)
                for model in args.models
            ]
            for fut in futures:
                results.append(fut.result())

    print("\n=== SUMMARY ===")
    failures = 0
    for model, code in results:
        primary_path, _ = build_output_paths(args.output_dir, model, timestamp)
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        if code != 0:
            failures += 1
        print(f"  {model:24s} {status:18s} -> {primary_path}")

    if failures:
        sys.exit(f"\n{failures}/{len(results)} run(s) failed.")
    print("\nAll runs completed. Outputs share timestamp "
          f"{timestamp} and can be aligned by diff_id.")


if __name__ == "__main__":
    main()
