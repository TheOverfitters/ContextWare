"""Build a blind, multi-label stratified sample for manual gold annotation.

The ambiguity study (PDF section 4) validates the three cloud LLMs against a
human gold standard built on a *stratified* sample that proportionally covers
all 16 categories. This script produces that sample.

Strategy
--------
The dataset is multi-label: every diff carries the set of categories scored
>= ``--threshold`` (0.5) by the primary extraction model (gemma). Because
labels co-occur (~2.9 per diff), strata overlap, so plain per-stratum
sampling does not work. Instead we:

  1. **Census the rarest labels** (``--census`` , default UI/UX + Performance):
     take *every* diff that carries one of them, since they are too rare to
     sample reliably.
  2. **Iterative stratification** (Sechidis et al. 2011) over the rest, to
     reach ``--size`` diffs while preserving each label's proportion across
     all 16 categories simultaneously.

gemma's labels are used ONLY to stratify — they are NOT the gold. The output
sheet is *blind*: it contains the diff content but none of the model
predictions, so annotators are not anchored.

Outputs (under ``--out-dir``)
-----------------------------
  * ``gold_sample_blind.csv``   - one row per diff: diff_id, filename,
    commit_url, commit_message, changed_lines, and an empty ``labels_gold``
    column to fill in (one column holding ALL applicable categories, primary
    included).
  * ``gold_sample_blind.jsonl`` - same, machine-readable (``labels_gold: []``).
  * ``gold_sample_key.jsonl``   - NOT for annotators: records the gemma label
    set / primary / census flag per diff, plus the RNG seed, for
    reproducibility and later scoring.

Example
-------
    python build_gold_sample.py --size 150 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

# This phase lives in its own folder but reuses the acf-analysis helpers and
# the extraction outputs, so we resolve those sibling paths explicitly.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ACF_ANALYSIS_DIR = REPO_ROOT / "acf-analysis"
sys.path.insert(0, str(ACF_ANALYSIS_DIR))
from acf_io import load_diffs_from_acf_json  # noqa: E402

DEFAULT_GEMMA_DIR = REPO_ROOT / "acf-outputs" / "gemma4_31b-cloud"
DEFAULT_DIFFS = REPO_ROOT / "outputs" / "git_history_diffs.json"
DEFAULT_OUT_DIR = SCRIPT_DIR / "gold-sample"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--gemma-file",
        type=Path,
        default=None,
        help="primary_*.jsonl from the extraction model (default: latest in "
             f"{DEFAULT_GEMMA_DIR}).",
    )
    p.add_argument("--diffs-json", type=Path, default=DEFAULT_DIFFS,
                   help=f"Source diffs (raw text), default {DEFAULT_DIFFS}.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory, default {DEFAULT_OUT_DIR}.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Score >= threshold marks a category as present (default 0.5).")
    p.add_argument("--size", type=int, default=150,
                   help="Target number of diffs in the sample (default 150).")
    p.add_argument("--census", nargs="*", default=["UI/UX", "Performance"],
                   help="Labels whose every diff is taken in full (default: UI/UX Performance).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    return p.parse_args()


def find_latest_primary(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    cands = sorted(model_dir.glob("primary_*.jsonl"), key=lambda x: x.stem, reverse=True)
    return cands[0] if cands else None


def cleaned_changed_lines(diff_text: str) -> str:
    """Return the added/removed lines, stripped of diff markers, one per line."""
    out: list[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+"):
            body = line[1:].strip()
            if body:
                out.append(f"+ {body}")
        elif line.startswith("-"):
            body = line[1:].strip()
            if body:
                out.append(f"- {body}")
    return "\n".join(out)


def label_set(scores: dict[str, float], threshold: float) -> list[str]:
    return sorted(c for c, v in scores.items() if isinstance(v, (int, float)) and v >= threshold)


def commit_file_url(repo: str, commit_hash: str, filename: str) -> str:
    """Link to a commit, anchored on the diff of *filename* only.

    GitHub has no standalone "single file inside a commit" page, but it anchors
    each file's diff as ``#diff-<sha256(path)>``, so the link opens the commit
    and jumps straight to the analysed file. If the anchor ever fails to match
    (e.g. GitHub changes the scheme) it degrades gracefully to the commit top.
    """
    if not (repo and commit_hash):
        return ""
    url = f"https://github.com/{repo}/commit/{commit_hash}"
    if filename:
        anchor = hashlib.sha256(filename.encode("utf-8")).hexdigest()
        url += f"#diff-{anchor}"
    return url


def write_xlsx_from_csv(csv_path: Path, xlsx_path: Path) -> bool:
    """Render the blind CSV as a real .xlsx so it opens cleanly in Excel.

    Solves the two Excel pain points of a raw CSV: locale-dependent delimiter
    (Italian Excel splits on ';', not ',') and multi-line cells. Returns False
    if openpyxl is not installed (the CSV is still produced).
    """
    try:
        from openpyxl import Workbook
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
        from openpyxl.styles import Alignment, Font
    except ImportError:
        print("[xlsx] openpyxl not installed; skipping .xlsx "
              "(install with: pip install openpyxl)", file=sys.stderr)
        return False

    # Diff text can carry control characters that are illegal in the XLSX XML;
    # Excel flags the file as corrupt and "repairs" it. Strip them up front.
    def clean(value: str) -> str:
        return ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "gold_sample"

    header_font = Font(bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    top = Alignment(vertical="top")

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=clean(value))
            if r_idx == 1:
                cell.font = header_font
            else:
                # changed_lines (5) and commit_message (4) can be long → wrap.
                cell.alignment = wrap if c_idx in (4, 5) else top

    # diff_id, filename, commit_url, commit_message, changed_lines, labels_gold
    widths = {1: 42, 2: 20, 3: 60, 4: 40, 5: 90, 6: 26}
    for col, width in widths.items():
        ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = width

    ws.freeze_panes = "A2"  # keep the header visible while scrolling
    wb.save(xlsx_path)
    return True


def iterative_stratification(
    pool: list[str],
    labels: dict[str, list[str]],
    size: int,
    rng: random.Random,
) -> list[str]:
    """Single-subset iterative stratification (Sechidis et al. 2011).

    Greedily pick ``size`` ids from ``pool`` so each label's proportion is
    preserved as closely as possible. At each step we serve the label that is
    most "starved" (smallest remaining desired count), breaking ties toward the
    rarest label, then take the diff that best fills the still-wanted labels.
    """
    pool_set = set(pool)
    if size <= 0 or not pool_set:
        return []
    if size >= len(pool_set):
        return list(pool_set)

    ratio = size / len(pool_set)
    label_total: Counter[str] = Counter()
    for i in pool_set:
        for lab in labels[i]:
            label_total[lab] += 1
    desired: dict[str, float] = {lab: ratio * tot for lab, tot in label_total.items()}

    selected: list[str] = []
    while len(selected) < size and pool_set:
        wanted = [lab for lab in desired if desired[lab] > 0 and label_total[lab] > 0]
        if not wanted:
            # No label still needs samples: fill the remainder uniformly.
            rest = list(pool_set)
            rng.shuffle(rest)
            for i in rest[: size - len(selected)]:
                selected.append(i)
                pool_set.discard(i)
            break

        min_des = min(desired[lab] for lab in wanted)
        tied = [lab for lab in wanted if desired[lab] == min_des]
        min_tot = min(label_total[lab] for lab in tied)
        tied = [lab for lab in tied if label_total[lab] == min_tot]
        lab = rng.choice(sorted(tied))

        cands = [i for i in pool_set if lab in labels[i]]
        rng.shuffle(cands)  # randomise ties under the max() below
        best = max(cands, key=lambda i: sum(max(0.0, desired[l]) for l in labels[i]))

        selected.append(best)
        pool_set.discard(best)
        for l in labels[best]:
            desired[l] -= 1
            label_total[l] -= 1

    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    gemma_file = args.gemma_file or find_latest_primary(DEFAULT_GEMMA_DIR)
    if not gemma_file or not gemma_file.exists():
        sys.exit(f"Extraction file not found (looked in {DEFAULT_GEMMA_DIR}).")
    if not args.diffs_json.exists():
        sys.exit(f"Source diffs not found: {args.diffs_json}")

    # 1. gemma label sets (stratification variable only).
    labels: dict[str, list[str]] = {}
    gemma_primary: dict[str, str] = {}
    for line in gemma_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        did = rec.get("diff_id")
        if not did:
            continue
        labels[did] = label_set(rec.get("category_scores", {}), args.threshold)
        gemma_primary[did] = (rec.get("primary") or {}).get("primary_category", "")

    # 2. raw diff text + commit message, joined by diff_id.
    src = {d["diff_id"]: d for d in load_diffs_from_acf_json(args.diffs_json)}

    # Drop diffs with no label above threshold or no source text: nothing to stratify on.
    usable = [d for d in labels if labels[d] and d in src]
    dropped = len(labels) - len(usable)
    if dropped:
        print(f"[info] skipped {dropped} diff(s) with empty label set or missing source text")

    # 3a. Census the rarest labels.
    census_set = set(args.census)
    censused = [d for d in usable if census_set.intersection(labels[d])]
    censused_ids = set(censused)
    if len(censused) > args.size:
        sys.exit(
            f"Census of {sorted(census_set)} already yields {len(censused)} diffs "
            f"> --size {args.size}. Raise --size or narrow --census."
        )

    # 3b. Iterative stratification on the rest.
    remaining_pool = [d for d in usable if d not in censused_ids]
    need = args.size - len(censused)
    strat = iterative_stratification(remaining_pool, labels, need, rng)

    selected = censused + strat
    rng.shuffle(selected)  # mix census + stratified so the sheet order is not informative

    # 4. Coverage report.
    cov = Counter()
    for d in selected:
        for lab in labels[d]:
            cov[lab] += 1
    all_labels = sorted({lab for ls in labels.values() for lab in ls})
    print(f"\n[sample] gemma file : {gemma_file.name}")
    print(f"[sample] threshold  : {args.threshold}")
    print(f"[sample] size       : {len(selected)} "
          f"(census={len(censused)} from {sorted(census_set)}, stratified={len(strat)})")
    print(f"[sample] seed       : {args.seed}")
    print("\n[coverage] label-instances in the sample (>= threshold):")
    for lab in sorted(all_labels, key=lambda x: cov[x], reverse=True):
        print(f"  {lab:22s} {cov[lab]:4d}")

    # 5. Write outputs.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blind_csv = args.out_dir / "gold_sample_blind.csv"
    blind_jsonl = args.out_dir / "gold_sample_blind.jsonl"
    key_jsonl = args.out_dir / "gold_sample_key.jsonl"

    with (
        blind_csv.open("w", encoding="utf-8-sig", newline="") as fcsv,
        blind_jsonl.open("w", encoding="utf-8") as fjsonl,
        key_jsonl.open("w", encoding="utf-8") as fkey,
    ):
        writer = csv.writer(fcsv)
        writer.writerow(["diff_id", "filename", "commit_url", "commit_message", "changed_lines", "labels_gold"])
        for did in selected:
            d = src[did]
            changed = cleaned_changed_lines(d.get("diff_text", ""))
            filename = d.get("filename", "")
            commit_message = (d.get("commit_message", "") or "").strip()
            repo = (d.get("repo", "") or "").strip()
            commit_hash = (d.get("commit_hash", "") or "").strip()
            commit_url = commit_file_url(repo, commit_hash, filename)
            # Blind sheet: labels_gold is intentionally empty for the annotator.
            writer.writerow([did, filename, commit_url, commit_message, changed, ""])
            fjsonl.write(json.dumps({
                "diff_id": did,
                "filename": filename,
                "commit_url": commit_url,
                "commit_message": commit_message,
                "changed_lines": changed,
                "labels_gold": [],
            }, ensure_ascii=False) + "\n")
            fkey.write(json.dumps({
                "diff_id": did,
                "gemma_labels": labels[did],
                "gemma_primary": gemma_primary.get(did, ""),
                "in_census": did in censused_ids,
                "seed": args.seed,
            }, ensure_ascii=False) + "\n")

    # Excel-friendly copy derived from the CSV we just wrote.
    blind_xlsx = args.out_dir / "gold_sample_blind.xlsx"
    xlsx_ok = write_xlsx_from_csv(blind_csv, blind_xlsx)

    print(f"\n[+] blind sheet (CSV)  : {blind_csv}")
    if xlsx_ok:
        print(f"[+] blind sheet (XLSX) : {blind_xlsx}")
    print(f"[+] blind sheet (JSONL): {blind_jsonl}")
    print(f"[+] key (NOT for annotators): {key_jsonl}")
    print("\nFill the 'labels_gold' column with all applicable categories "
          "(primary included), e.g. \"Testing; Documentation\".")


if __name__ == "__main__":
    main()
