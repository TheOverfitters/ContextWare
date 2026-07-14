"""Inter-model agreement & ambiguity analysis for the ACF classifier study.

This consumes the per-model classifier outputs under ``acf-outputs/<model>/``
(written by ``acf_analyser.py`` / ``run_multi_model.py``) and computes the
inter-rater agreement metrics for section 5 of the study, treating each of the
three cloud LLMs as an independent *rater* of the same diffs:

  1. Single-label agreement on ``primary_category``
       - % full agreement (all 3 raters) and mean pairwise % agreement
       - Fleiss' kappa (3 raters, k categories)
       - Cohen's kappa for each rater pair (+ mean)
       - Krippendorff's alpha (nominal)

  2. Multi-label agreement on the above-threshold category *sets*
     (categories with ``category_scores >= threshold``)
       - mean pairwise Jaccard similarity (per instance, averaged)
       - per category: Fleiss' kappa, Krippendorff's alpha, mean pairwise
         Cohen kappa (each category as a binary present/absent label)

  3. Ambiguity score per instance and aggregated per category (the key §5
     result -- "which categories are most ambiguous")
       - per-instance: Shannon entropy of the 3 primary votes + disagreement
         fraction; multi-label mean pairwise Jaccard *distance*
       - per category: presence disagreement rate (fraction of instances where
         the 3 raters are not unanimous about that category) and mean primary
         vote entropy among instances whose plurality primary is that category

All metrics are implemented in the standard library so the formulas stay
auditable. Raters are aligned on the diffs classified by *all* models.

Examples
--------
Run with defaults (three study models, threshold 0.5)::

    python agreement_analysis.py

Pin a single shared run and a custom threshold::

    python agreement_analysis.py --timestamp 20260619_180651 --threshold 0.6
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

# Reuse the analyser's model->directory mapping so the dir names match exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acf_io import sanitize_model_name  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# The three cloud LLMs used for the ambiguity study (same as run_multi_model.py).
DEFAULT_MODELS = [
    "kimi-k2.7-code:cloud",
    "qwen3.5:cloud",
    "glm-5.2:cloud",
]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "acf-outputs"
DEFAULT_CATEGORIES_PATH = SCRIPT_DIR.parent / "categories.json"
# The primary LLM the study models are benchmarked against (treated as reference).
DEFAULT_REFERENCE = "gemma4:31b-cloud"

# Maintenance-reason axis (ISO/IEC/IEEE 14764). Two label spaces are scored:
# the base type (5, adaptive leaves collapsed) and the parent class (2).
MAINT_BASE_ORDER = ["corrective", "preventive", "adaptive", "additive", "perfective"]
MAINT_CLASS_ORDER = ["Correction", "Enhancement"]


def load_category_names(path: Path) -> list[str]:
    """Read the ordered category names from categories.json.

    Accepts either ``{"categories": [{"name": ...}, ...]}`` or a bare list of
    names/objects. Returns [] when the file is missing or unparseable so the
    caller can fall back.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("categories", data) if isinstance(data, dict) else data
    names: list[str] = []
    for it in items or []:
        name = it.get("name") if isinstance(it, dict) else it
        if isinstance(name, str) and name:
            names.append(name)
    return names


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_model_records(
    output_dir: Path,
    model: str,
    timestamp: str | None,
) -> dict[str, dict]:
    """Return ``{diff_id: record}`` for one model.

    When *timestamp* is given only ``primary_<timestamp>.jsonl`` is read;
    otherwise every ``primary_*.jsonl`` is merged with later runs (sorted by
    filename, i.e. by timestamp) overwriting earlier ones per ``diff_id``.
    """
    model_dir = output_dir / sanitize_model_name(model)
    if timestamp:
        files = [model_dir / f"primary_{timestamp}.jsonl"]
    else:
        files = sorted(model_dir.glob("primary_*.jsonl"))

    by_id: dict[str, dict] = {}
    for fn in files:
        if not fn.exists():
            continue
        with fn.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                diff_id = str(rec.get("diff_id", ""))
                if diff_id:
                    by_id[diff_id] = rec
    return by_id


def above_threshold_set(record: dict, categories: list[str], threshold: float) -> frozenset[str]:
    """Multi-label set: categories whose score is >= *threshold*."""
    scores = record.get("category_scores") or {}
    return frozenset(c for c in categories if float(scores.get(c, 0.0)) >= threshold)


def primary_of(record: dict) -> str:
    primary = record.get("primary") or {}
    return str(primary.get("primary_category", "") or "")


def maint_base_of(record: dict) -> str:
    """Primary maintenance base type (adaptive leaves collapsed); "" if absent."""
    m = record.get("maintenance") or {}
    return str(m.get("maintenance_base_type") or "")


def maint_class_of(record: dict) -> str:
    """Primary maintenance parent class (Correction/Enhancement); "" if absent."""
    m = record.get("maintenance") or {}
    return str(m.get("maintenance_class") or "")


def has_maintenance_axis(aligned: dict[str, list[dict]]) -> bool:
    """True when at least one aligned record carries a maintenance base type."""
    return any(
        maint_base_of(r)
        for rows in aligned.values()
        for r in rows
    )


# --------------------------------------------------------------------------- #
# Agreement metrics (pure stdlib)
# --------------------------------------------------------------------------- #
def percent_full_agreement(rows: list[list[str]]) -> float:
    """Fraction of items where all raters chose the same label."""
    if not rows:
        return float("nan")
    agree = sum(1 for r in rows if len(set(r)) == 1)
    return agree / len(rows)


def pairwise_percent_agreement(rows: list[list[str]]) -> dict[tuple[int, int], float]:
    """Observed agreement for each rater pair (column index pair)."""
    out: dict[tuple[int, int], float] = {}
    n_raters = len(rows[0]) if rows else 0
    for a, b in combinations(range(n_raters), 2):
        same = sum(1 for r in rows if r[a] == r[b])
        out[(a, b)] = same / len(rows) if rows else float("nan")
    return out


def cohen_kappa(col_a: list[str], col_b: list[str]) -> float:
    """Cohen's kappa for two raters over a shared item set."""
    n = len(col_a)
    if n == 0:
        return float("nan")
    labels = set(col_a) | set(col_b)
    p_o = sum(1 for x, y in zip(col_a, col_b) if x == y) / n
    count_a = Counter(col_a)
    count_b = Counter(col_b)
    p_e = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else float("nan")
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(rows: list[list[str]], labels: list[str]) -> float:
    """Fleiss' kappa for a fixed number of raters per item.

    *rows* is one list of rater labels per item; *labels* is the category space.
    """
    n_items = len(rows)
    if n_items == 0:
        return float("nan")
    n_raters = len(rows[0])
    if n_raters < 2:
        return float("nan")
    label_index = {l: j for j, l in enumerate(labels)}

    # n_ij = number of raters that assigned category j to item i.
    counts = [[0] * len(labels) for _ in range(n_items)]
    for i, row in enumerate(rows):
        for lab in row:
            counts[i][label_index[lab]] += 1

    # Per-item agreement P_i.
    p_i = []
    for row_counts in counts:
        s = sum(c * c for c in row_counts)
        p_i.append((s - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / n_items

    # Expected agreement from category marginals p_j.
    total = n_items * n_raters
    p_j = [sum(counts[i][j] for i in range(n_items)) / total for j in range(len(labels))]
    p_e = sum(p * p for p in p_j)

    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else float("nan")
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(rows: list[list[str]]) -> float:
    """Krippendorff's alpha (nominal metric) from a units x raters matrix.

    Built from the coincidence matrix; assumes every unit has the same number
    of raters (no missing values), which holds for the aligned diff set.
    """
    if not rows:
        return float("nan")
    m = len(rows[0])  # raters per unit
    if m < 2:
        return float("nan")

    # Observed coincidences o_ck over all ordered value pairs within each unit.
    o: Counter[tuple[str, str]] = Counter()
    for row in rows:
        weight = 1.0 / (m - 1)
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                o[(row[a], row[b])] += weight

    n_c: Counter[str] = Counter()
    for (c, _k), v in o.items():
        n_c[c] += v
    n_total = sum(n_c.values())
    if n_total <= 1:
        return float("nan")

    d_o = sum(v for (c, k), v in o.items() if c != k)
    d_e = sum(
        n_c[c] * n_c[k]
        for c in n_c
        for k in n_c
        if c != k
    ) / (n_total - 1)
    if d_e == 0:
        return 1.0 if d_o == 0 else float("nan")
    return 1 - d_o / d_e


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity; two empty sets count as full agreement (1.0)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def shannon_entropy_bits(votes: list[str]) -> float:
    """Shannon entropy (bits) of the distribution of *votes*."""
    n = len(votes)
    if n == 0:
        return 0.0
    counts = Counter(votes)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return 0.0 if ent == 0 else ent  # avoid -0.0 from the negated sum


def plurality(votes: list[str]) -> str:
    """Most common vote; ties broken by alphabetical order for determinism."""
    counts = Counter(votes)
    top = max(counts.values())
    return sorted(c for c, n in counts.items() if n == top)[0]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _single_label_block(
    rows: list[list[str]],
    model_names: list[str],
    label_space: list[str],
) -> dict:
    """Single-label agreement metrics over a units x raters matrix of labels.

    Shared by the category ``primary_category`` axis and the maintenance-reason
    axes (base type and parent class): each is one nominal label per rater.
    """
    n_raters = len(model_names)
    pair_pct = pairwise_percent_agreement(rows)
    pair_cohen = {
        (a, b): cohen_kappa([r[a] for r in rows], [r[b] for r in rows])
        for a, b in combinations(range(n_raters), 2)
    }
    return {
        "percent_full_agreement": percent_full_agreement(rows),
        "mean_pairwise_percent_agreement": _mean(list(pair_pct.values())),
        "fleiss_kappa": fleiss_kappa(rows, label_space),
        "krippendorff_alpha": krippendorff_alpha_nominal(rows),
        "pairwise_percent_agreement": {
            f"{model_names[a]} vs {model_names[b]}": v for (a, b), v in pair_pct.items()
        },
        "pairwise_cohen_kappa": {
            f"{model_names[a]} vs {model_names[b]}": v for (a, b), v in pair_cohen.items()
        },
        "mean_pairwise_cohen_kappa": _mean(list(pair_cohen.values())),
    }


def analyse(
    aligned: dict[str, list[dict]],
    model_names: list[str],
    categories: list[str],
    threshold: float,
) -> dict:
    diff_ids = list(aligned.keys())
    n_raters = len(model_names)

    # ---- single-label primary_category ------------------------------------ #
    primary_rows = [[primary_of(r) for r in aligned[d]] for d in diff_ids]
    # Label space = declared categories plus any observed value (e.g. "" for a
    # missing primary) so Fleiss' kappa never hits an unknown label.
    primary_labels = sorted(set(categories) | {lab for row in primary_rows for lab in row})

    pct_full = percent_full_agreement(primary_rows)
    pair_pct = pairwise_percent_agreement(primary_rows)
    pair_cohen = {
        (a, b): cohen_kappa([row[a] for row in primary_rows], [row[b] for row in primary_rows])
        for a, b in combinations(range(n_raters), 2)
    }
    single = {
        "percent_full_agreement": pct_full,
        "mean_pairwise_percent_agreement": _mean(list(pair_pct.values())),
        "fleiss_kappa": fleiss_kappa(primary_rows, primary_labels),
        "krippendorff_alpha": krippendorff_alpha_nominal(primary_rows),
        "pairwise_percent_agreement": {
            f"{model_names[a]} vs {model_names[b]}": v for (a, b), v in pair_pct.items()
        },
        "pairwise_cohen_kappa": {
            f"{model_names[a]} vs {model_names[b]}": v for (a, b), v in pair_cohen.items()
        },
        "mean_pairwise_cohen_kappa": _mean(list(pair_cohen.values())),
    }

    # ---- multi-label above-threshold sets --------------------------------- #
    sets_by_diff = {
        d: [above_threshold_set(r, categories, threshold) for r in aligned[d]]
        for d in diff_ids
    }

    # Mean pairwise Jaccard per instance, then averaged over instances.
    instance_jaccard = {}
    for d in diff_ids:
        sets = sets_by_diff[d]
        pair_j = [jaccard(sets[a], sets[b]) for a, b in combinations(range(n_raters), 2)]
        instance_jaccard[d] = _mean(pair_j)
    mean_jaccard = _mean(list(instance_jaccard.values()))

    # Per-category binary agreement (present/absent under threshold).
    per_category = {}
    for cat in categories:
        bin_rows = [
            [("1" if cat in s else "0") for s in sets_by_diff[d]]
            for d in diff_ids
        ]
        # Prevalence = mean presence over all rater-judgements.
        present = sum(r.count("1") for r in bin_rows)
        prevalence = present / (len(bin_rows) * n_raters) if bin_rows else float("nan")
        # Disagreement rate: fraction of instances where raters are not unanimous.
        disagree = sum(1 for r in bin_rows if len(set(r)) > 1) / len(bin_rows) if bin_rows else float("nan")
        pair_ck = [
            cohen_kappa([row[a] for row in bin_rows], [row[b] for row in bin_rows])
            for a, b in combinations(range(n_raters), 2)
        ]
        per_category[cat] = {
            "prevalence": prevalence,
            "presence_disagreement_rate": disagree,
            "fleiss_kappa": fleiss_kappa(bin_rows, ["0", "1"]),
            "krippendorff_alpha": krippendorff_alpha_nominal(bin_rows),
            "mean_pairwise_cohen_kappa": _mean(pair_ck),
        }

    multi = {
        "threshold": threshold,
        "mean_pairwise_jaccard": mean_jaccard,
        "per_category": per_category,
    }

    # ---- ambiguity per instance + aggregated per category ----------------- #
    log2_r = math.log2(n_raters) if n_raters > 1 else 1.0
    per_instance = []
    # Accumulators for per-category ambiguity aggregated by plurality primary.
    entropy_by_primary: dict[str, list[float]] = {c: [] for c in categories}
    for d in diff_ids:
        votes = [primary_of(r) for r in aligned[d]]
        ent = shannon_entropy_bits(votes)
        norm_ent = ent / log2_r if log2_r else 0.0
        # Disagreement fraction: 1 - (size of largest agreeing bloc)/n_raters.
        disagree_frac = 1 - (max(Counter(votes).values()) / n_raters)
        plur = plurality(votes)
        jdist = 1 - instance_jaccard[d]  # multi-label disagreement
        entropy_by_primary.setdefault(plur, []).append(norm_ent)
        per_instance.append(
            {
                "diff_id": d,
                "primary_votes": "|".join(votes),
                "plurality_primary": plur,
                "primary_entropy_bits": ent,
                "primary_entropy_norm": norm_ent,
                "primary_disagreement_fraction": disagree_frac,
                "multilabel_jaccard_distance": jdist,
            }
        )

    # Per-category ambiguity table = key §5 result.
    category_ambiguity = {}
    for cat in categories:
        ents = entropy_by_primary.get(cat, [])
        category_ambiguity[cat] = {
            "n_as_plurality_primary": len(ents),
            "mean_primary_entropy_norm": _mean(ents) if ents else float("nan"),
            "presence_disagreement_rate": per_category[cat]["presence_disagreement_rate"],
            "fleiss_kappa": per_category[cat]["fleiss_kappa"],
            "prevalence": per_category[cat]["prevalence"],
        }

    ambiguity = {
        "mean_primary_entropy_norm": _mean([pi["primary_entropy_norm"] for pi in per_instance]),
        "mean_primary_disagreement_fraction": _mean(
            [pi["primary_disagreement_fraction"] for pi in per_instance]
        ),
        "mean_multilabel_jaccard_distance": _mean(
            [pi["multilabel_jaccard_distance"] for pi in per_instance]
        ),
        "category_ambiguity": category_ambiguity,
    }

    return {
        "n_items": len(diff_ids),
        "n_raters": n_raters,
        "raters": model_names,
        "single_label_primary": single,
        "multi_label_threshold": multi,
        "ambiguity": ambiguity,
        "_per_instance": per_instance,  # written to CSV, dropped from summary.json
    }


def analyse_vs_reference(
    aligned: dict[str, list[dict]],
    ref_records: dict[str, dict],
    model_names: list[str],
    ref_name: str,
    categories: list[str],
    threshold: float,
) -> dict:
    """Compare each study model against the reference LLM (gemma) one-to-one.

    Restricted to diffs classified by *all* study models and the reference, so
    the per-model numbers are directly comparable. Uses the same metric family:
    primary-label agreement / Cohen kappa and multi-label Jaccard + per-category
    Cohen kappa.
    """
    diff_ids = [d for d in aligned if d in ref_records]
    ref_primary = {d: primary_of(ref_records[d]) for d in diff_ids}
    ref_set = {d: above_threshold_set(ref_records[d], categories, threshold) for d in diff_ids}

    per_model: dict[str, dict] = {}
    for mi, model in enumerate(model_names):
        study_primary = [primary_of(aligned[d][mi]) for d in diff_ids]
        ref_col = [ref_primary[d] for d in diff_ids]
        study_sets = {d: above_threshold_set(aligned[d][mi], categories, threshold) for d in diff_ids}

        primary_agreement = _mean([1.0 if a == b else 0.0 for a, b in zip(study_primary, ref_col)])
        primary_kappa = cohen_kappa(study_primary, ref_col)
        mean_jacc = _mean([jaccard(study_sets[d], ref_set[d]) for d in diff_ids])

        per_cat_kappa = {}
        for cat in categories:
            a = ["1" if cat in study_sets[d] else "0" for d in diff_ids]
            b = ["1" if cat in ref_set[d] else "0" for d in diff_ids]
            per_cat_kappa[cat] = cohen_kappa(a, b)

        per_model[model] = {
            "primary_agreement": primary_agreement,
            "primary_cohen_kappa": primary_kappa,
            "mean_jaccard": mean_jacc,
            "mean_per_category_cohen_kappa": _mean(list(per_cat_kappa.values())),
            "per_category_cohen_kappa": per_cat_kappa,
        }

    return {
        "reference": ref_name,
        "n_items": len(diff_ids),
        "threshold": threshold,
        "per_model": per_model,
    }


def analyse_maintenance(
    aligned: dict[str, list[dict]],
    model_names: list[str],
) -> dict | None:
    """Inter-model agreement on the maintenance-reason axis (ISO/IEC/IEEE 14764).

    Scores two nominal label spaces, both single-label (one dominant reason per
    diff): the base type (5, adaptive leaves collapsed) and the parent class
    (Correction/Enhancement). Also reports, per base type, how ambiguous it is
    across raters (presence-disagreement + binary kappa), the analogue of the
    per-category ambiguity table. Returns ``None`` on legacy inputs with no
    maintenance block.
    """
    if not has_maintenance_axis(aligned):
        return None

    diff_ids = list(aligned.keys())
    n_raters = len(model_names)

    base_rows = [[maint_base_of(r) for r in aligned[d]] for d in diff_ids]
    class_rows = [[maint_class_of(r) for r in aligned[d]] for d in diff_ids]

    base_labels = sorted({lab for row in base_rows for lab in row} | set(MAINT_BASE_ORDER))
    class_labels = sorted({lab for row in class_rows for lab in row} | set(MAINT_CLASS_ORDER))

    base_block = _single_label_block(base_rows, model_names, base_labels)
    class_block = _single_label_block(class_rows, model_names, class_labels)

    # Per-type ambiguity: treat "this rater's primary base type == t" as a binary
    # present/absent label, so we can see which reason drives disagreement.
    observed = {lab for row in base_rows for lab in row if lab}
    type_order = [t for t in MAINT_BASE_ORDER if t in observed] + sorted(observed - set(MAINT_BASE_ORDER))
    per_type = {}
    for t in type_order:
        bin_rows = [[("1" if lab == t else "0") for lab in row] for row in base_rows]
        present = sum(r.count("1") for r in bin_rows)
        prevalence = present / (len(bin_rows) * n_raters) if bin_rows else float("nan")
        disagree = sum(1 for r in bin_rows if len(set(r)) > 1) / len(bin_rows) if bin_rows else float("nan")
        pair_ck = [
            cohen_kappa([row[a] for row in bin_rows], [row[b] for row in bin_rows])
            for a, b in combinations(range(n_raters), 2)
        ]
        per_type[t] = {
            "prevalence": prevalence,
            "presence_disagreement_rate": disagree,
            "fleiss_kappa": fleiss_kappa(bin_rows, ["0", "1"]),
            "mean_pairwise_cohen_kappa": _mean(pair_ck),
        }

    # Per-instance rows for CSV / auditing.
    log2_r = math.log2(n_raters) if n_raters > 1 else 1.0
    per_instance = []
    for i, d in enumerate(diff_ids):
        votes = base_rows[i]
        cvotes = class_rows[i]
        ent = shannon_entropy_bits(votes)
        per_instance.append({
            "diff_id": d,
            "base_votes": "|".join(votes),
            "plurality_base": plurality(votes),
            "base_entropy_norm": ent / log2_r if log2_r else 0.0,
            "base_disagreement_fraction": 1 - (max(Counter(votes).values()) / n_raters),
            "class_votes": "|".join(cvotes),
            "class_full_agreement": 1.0 if len(set(cvotes)) == 1 else 0.0,
        })

    return {
        "n_items": len(diff_ids),
        "n_raters": n_raters,
        "raters": model_names,
        "base_type": base_block,
        "parent_class": class_block,
        "per_type": per_type,
        "_per_instance": per_instance,
    }


def analyse_maintenance_vs_reference(
    aligned: dict[str, list[dict]],
    ref_records: dict[str, dict],
    model_names: list[str],
    ref_name: str,
) -> dict | None:
    """Compare each study model's maintenance reason against the reference LLM.

    Reports, per study model, primary-agreement + Cohen kappa on both the base
    type and the parent class, on the diffs shared with the reference. Returns
    ``None`` when neither side carries a maintenance block.
    """
    diff_ids = [d for d in aligned if d in ref_records]
    if not diff_ids:
        return None
    if not has_maintenance_axis({d: aligned[d] for d in diff_ids}) and not any(
        maint_base_of(ref_records[d]) for d in diff_ids
    ):
        return None

    ref_base = {d: maint_base_of(ref_records[d]) for d in diff_ids}
    ref_class = {d: maint_class_of(ref_records[d]) for d in diff_ids}

    per_model: dict[str, dict] = {}
    for mi, model in enumerate(model_names):
        s_base = [maint_base_of(aligned[d][mi]) for d in diff_ids]
        s_class = [maint_class_of(aligned[d][mi]) for d in diff_ids]
        r_base = [ref_base[d] for d in diff_ids]
        r_class = [ref_class[d] for d in diff_ids]
        per_model[model] = {
            "base_agreement": _mean([1.0 if a == b else 0.0 for a, b in zip(s_base, r_base)]),
            "base_cohen_kappa": cohen_kappa(s_base, r_base),
            "class_agreement": _mean([1.0 if a == b else 0.0 for a, b in zip(s_class, r_class)]),
            "class_cohen_kappa": cohen_kappa(s_class, r_class),
        }

    return {
        "reference": ref_name,
        "n_items": len(diff_ids),
        "per_model": per_model,
    }


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return "  nan" if isinstance(x, float) and math.isnan(x) else f"{x:6.3f}"


def print_report(result: dict) -> None:
    s = result["single_label_primary"]
    m = result["multi_label_threshold"]
    a = result["ambiguity"]
    print("\n" + "=" * 72)
    print(f"INTER-MODEL AGREEMENT  |  {result['n_items']} diffs  |  raters: {', '.join(result['raters'])}")
    print("=" * 72)

    print("\n[1] SINGLE-LABEL  (primary_category)")
    print(f"    full agreement (all {result['n_raters']}) : {_fmt(s['percent_full_agreement'])}")
    print(f"    mean pairwise agreement     : {_fmt(s['mean_pairwise_percent_agreement'])}")
    print(f"    Fleiss' kappa               : {_fmt(s['fleiss_kappa'])}")
    print(f"    Krippendorff's alpha        : {_fmt(s['krippendorff_alpha'])}")
    print(f"    mean pairwise Cohen kappa   : {_fmt(s['mean_pairwise_cohen_kappa'])}")
    for pair, v in s["pairwise_cohen_kappa"].items():
        print(f"        Cohen kappa {pair:32s}: {_fmt(v)}")

    print(f"\n[2] MULTI-LABEL  (scores >= {m['threshold']})")
    print(f"    mean pairwise Jaccard       : {_fmt(m['mean_pairwise_jaccard'])}")
    print(f"    {'category':18s} {'prev':>6s} {'disagr':>7s} {'fleissK':>8s} {'alpha':>7s} {'cohenK':>7s}")
    for cat, c in sorted(m["per_category"].items(), key=lambda kv: kv[1]["fleiss_kappa"] if not math.isnan(kv[1]["fleiss_kappa"]) else 1e9):
        print(
            f"    {cat:18s} {c['prevalence']:6.3f} {c['presence_disagreement_rate']:7.3f} "
            f"{_fmt(c['fleiss_kappa'])} {_fmt(c['krippendorff_alpha'])} {_fmt(c['mean_pairwise_cohen_kappa'])}"
        )

    print("\n[3] AMBIGUITY  (most ambiguous categories first)")
    print(f"    mean primary entropy (norm) : {_fmt(a['mean_primary_entropy_norm'])}")
    print(f"    mean primary disagreement   : {_fmt(a['mean_primary_disagreement_fraction'])}")
    print(f"    mean multilabel Jacc. dist. : {_fmt(a['mean_multilabel_jaccard_distance'])}")
    print(f"    {'category':18s} {'n_prim':>6s} {'entropy':>8s} {'presDisagr':>10s} {'fleissK':>8s}")
    ranked = sorted(
        a["category_ambiguity"].items(),
        key=lambda kv: kv[1]["presence_disagreement_rate"]
        if not math.isnan(kv[1]["presence_disagreement_rate"]) else -1,
        reverse=True,
    )
    for cat, c in ranked:
        print(
            f"    {cat:18s} {c['n_as_plurality_primary']:6d} "
            f"{_fmt(c['mean_primary_entropy_norm'])} {c['presence_disagreement_rate']:10.3f} "
            f"{_fmt(c['fleiss_kappa'])}"
        )

    ref = result.get("reference_comparison")
    if ref:
        print(f"\n[4] vs REFERENCE LLM  ({ref['reference']}, {ref['n_items']} diffs, threshold {ref['threshold']})")
        print(f"    {'model':24s} {'primAgr':>8s} {'primK':>7s} {'Jaccard':>8s} {'catK':>7s}")
        for model, c in ref["per_model"].items():
            print(
                f"    {model:24s} {c['primary_agreement']:8.3f} {_fmt(c['primary_cohen_kappa'])} "
                f"{c['mean_jaccard']:8.3f} {_fmt(c['mean_per_category_cohen_kappa'])}"
            )

    maint = result.get("maintenance_agreement")
    if maint:
        print(f"\n[5] MAINTENANCE REASON  (ISO/IEC/IEEE 14764, {maint['n_items']} diffs)")
        for level_key, level_label in (
            ("base_type", "base type (5): corrective/preventive/adaptive/additive/perfective"),
            ("parent_class", "parent class (2): Correction / Enhancement"),
        ):
            b = maint[level_key]
            print(f"    -- {level_label} --")
            print(f"       full agreement (all {maint['n_raters']}) : {_fmt(b['percent_full_agreement'])}")
            print(f"       mean pairwise agreement     : {_fmt(b['mean_pairwise_percent_agreement'])}")
            print(f"       Fleiss' kappa               : {_fmt(b['fleiss_kappa'])}")
            print(f"       Krippendorff's alpha        : {_fmt(b['krippendorff_alpha'])}")
            print(f"       mean pairwise Cohen kappa   : {_fmt(b['mean_pairwise_cohen_kappa'])}")
        print("    per-type ambiguity (most ambiguous first):")
        print(f"    {'type':14s} {'prev':>6s} {'disagr':>7s} {'fleissK':>8s} {'cohenK':>7s}")
        for t, c in sorted(
            maint["per_type"].items(),
            key=lambda kv: kv[1]["presence_disagreement_rate"]
            if not math.isnan(kv[1]["presence_disagreement_rate"]) else -1,
            reverse=True,
        ):
            print(
                f"    {t:14s} {c['prevalence']:6.3f} {c['presence_disagreement_rate']:7.3f} "
                f"{_fmt(c['fleiss_kappa'])} {_fmt(c['mean_pairwise_cohen_kappa'])}"
            )

        mref = maint.get("reference_comparison")
        if mref:
            print(f"\n[6] MAINTENANCE vs REFERENCE  ({mref['reference']}, {mref['n_items']} diffs)")
            print(f"    {'model':24s} {'baseAgr':>8s} {'baseK':>7s} {'classAgr':>9s} {'classK':>7s}")
            for model, c in mref["per_model"].items():
                print(
                    f"    {model:24s} {c['base_agreement']:8.3f} {_fmt(c['base_cohen_kappa'])} "
                    f"{c['class_agreement']:9.3f} {_fmt(c['class_cohen_kappa'])}"
                )
    print("=" * 72)


def write_outputs(result: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    per_instance = result.pop("_per_instance")
    # Maintenance per-instance rows go to their own CSV, not the summary JSON.
    maint = result.get("maintenance_agreement")
    maint_per_instance = maint.pop("_per_instance", None) if isinstance(maint, dict) else None
    (report_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    # per-category table (multi-label + ambiguity merged).
    cats = result["multi_label_threshold"]["per_category"]
    amb = result["ambiguity"]["category_ambiguity"]
    with (report_dir / "per_category.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["category", "prevalence", "presence_disagreement_rate", "fleiss_kappa",
             "krippendorff_alpha", "mean_pairwise_cohen_kappa",
             "n_as_plurality_primary", "mean_primary_entropy_norm"]
        )
        for cat in cats:
            c = cats[cat]
            ac = amb[cat]
            w.writerow([
                cat, c["prevalence"], c["presence_disagreement_rate"], c["fleiss_kappa"],
                c["krippendorff_alpha"], c["mean_pairwise_cohen_kappa"],
                ac["n_as_plurality_primary"], ac["mean_primary_entropy_norm"],
            ])

    with (report_dir / "per_instance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_instance[0].keys()))
        w.writeheader()
        w.writerows(per_instance)

    ref = result.get("reference_comparison")
    if ref:
        # Per-model headline metrics vs the reference LLM.
        with (report_dir / "reference_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "reference", "n_items", "primary_agreement",
                        "primary_cohen_kappa", "mean_jaccard", "mean_per_category_cohen_kappa"])
            for model, c in ref["per_model"].items():
                w.writerow([model, ref["reference"], ref["n_items"], c["primary_agreement"],
                            c["primary_cohen_kappa"], c["mean_jaccard"], c["mean_per_category_cohen_kappa"]])
        # Per-category Cohen kappa of each study model vs the reference.
        with (report_dir / "reference_category_kappa.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            models = list(ref["per_model"])
            w.writerow(["category"] + models)
            cats = next(iter(ref["per_model"].values()))["per_category_cohen_kappa"].keys()
            for cat in cats:
                w.writerow([cat] + [ref["per_model"][m]["per_category_cohen_kappa"][cat] for m in models])

    if isinstance(maint, dict):
        # Per-type maintenance ambiguity table.
        with (report_dir / "maintenance_per_type.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["type", "prevalence", "presence_disagreement_rate",
                        "fleiss_kappa", "mean_pairwise_cohen_kappa"])
            for t, c in maint["per_type"].items():
                w.writerow([t, c["prevalence"], c["presence_disagreement_rate"],
                            c["fleiss_kappa"], c["mean_pairwise_cohen_kappa"]])
        # Per-instance maintenance votes.
        if maint_per_instance:
            with (report_dir / "maintenance_per_instance.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(maint_per_instance[0].keys()))
                w.writeheader()
                w.writerows(maint_per_instance)
        # Maintenance agreement vs the reference LLM, per study model.
        mref = maint.get("reference_comparison")
        if mref:
            with (report_dir / "maintenance_reference.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["model", "reference", "n_items", "base_agreement",
                            "base_cohen_kappa", "class_agreement", "class_cohen_kappa"])
                for model, c in mref["per_model"].items():
                    w.writerow([model, mref["reference"], mref["n_items"], c["base_agreement"],
                                c["base_cohen_kappa"], c["class_agreement"], c["class_cohen_kappa"]])

    print(f"\nWrote: {report_dir / 'summary.json'}")
    print(f"       {report_dir / 'per_category.csv'}")
    print(f"       {report_dir / 'per_instance.csv'}")


# --------------------------------------------------------------------------- #
# Charts (matplotlib; matches the repo's category_analysis.py conventions)
# --------------------------------------------------------------------------- #
def make_charts(result: dict, report_dir: Path) -> None:
    """Render the §5 figures into *report_dir*. No-op (with a note) if matplotlib
    is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: just write PNGs, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print("[charts] matplotlib not available; skipping figures.")
        return

    per_cat = result["multi_label_threshold"]["per_category"]
    amb = result["ambiguity"]["category_ambiguity"]
    single = result["single_label_primary"]
    threshold = result["multi_label_threshold"]["threshold"]

    # Minimum plurality-primary support below which a per-category entropy is too
    # noisy to rank on (e.g. n<=2 categories dominated the raw chart otherwise).
    MIN_SUPPORT = 20

    # (a) Key §5 result: categories ranked by presence-disagreement rate, with
    #     their Fleiss' kappa overlaid. Disagreement (0..~0.2) and kappa (0.4..0.8)
    #     are not commensurable, so each gets its own x-axis rather than sharing one.
    cats = sorted(per_cat, key=lambda c: per_cat[c]["presence_disagreement_rate"], reverse=True)
    disagree = [per_cat[c]["presence_disagreement_rate"] for c in cats]
    kappa = [per_cat[c]["fleiss_kappa"] for c in cats]
    y = list(range(len(cats)))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y, disagree, color="#d1495b")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()  # most ambiguous on top
    ax.set_xlabel("Presence disagreement rate", color="#d1495b")
    ax.tick_params(axis="x", colors="#d1495b")
    ax.set_xlim(0, max(disagree) * 1.15)
    ax.set_title(f"Category ambiguity (multi-label, threshold {threshold})")

    ax_k = ax.twiny()  # Fleiss' kappa on an independent top axis
    ax_k.plot(kappa, y, "o-", color="#1b3a4b")
    ax_k.set_xlabel("Fleiss' kappa (agreement)", color="#1b3a4b")
    ax_k.tick_params(axis="x", colors="#1b3a4b")
    ax_k.set_xlim(0, 1)

    ax.plot([], [], color="#d1495b", linewidth=6, label="presence disagreement rate")
    ax.plot([], [], "o-", color="#1b3a4b", label="Fleiss' kappa")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(plt, report_dir / "category_ambiguity.png")

    # (b) Mean primary-vote entropy per category (the entropy view of ambiguity).
    #     Low-support categories are shown greyed out: an entropy over 1-2 diffs is
    #     not a reliable ranking signal, so it must not read as a headline result.
    cats_e = sorted(amb, key=lambda c: (amb[c]["mean_primary_entropy_norm"]
                                        if not math.isnan(amb[c]["mean_primary_entropy_norm"]) else -1),
                    reverse=True)
    ent = [0.0 if math.isnan(amb[c]["mean_primary_entropy_norm"]) else amb[c]["mean_primary_entropy_norm"]
           for c in cats_e]
    n_prim = [amb[c]["n_as_plurality_primary"] for c in cats_e]
    colors = ["#edae49" if n >= MIN_SUPPORT else "#d9d9d9" for n in n_prim]
    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(cats_e)), ent, color=colors)
    for rect, n in zip(bars, n_prim):  # annotate with support (n diffs)
        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"n={n}", ha="center", va="bottom", fontsize=8)
    plt.xticks(range(len(cats_e)), cats_e, rotation=45, ha="right")
    plt.ylabel("Mean primary-vote entropy (normalised)")
    plt.title("Per-category disagreement entropy (when chosen as primary)")
    plt.plot([], [], color="#edae49", linewidth=6, label=f"n ≥ {MIN_SUPPORT}")
    plt.plot([], [], color="#d9d9d9", linewidth=6, label=f"n < {MIN_SUPPORT} (low support)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    _save(plt, report_dir / "category_entropy.png")

    # (c) Per-category agreement: Fleiss' kappa sorted ascending (worst first).
    cats_k = sorted(per_cat, key=lambda c: per_cat[c]["fleiss_kappa"])
    kvals = [per_cat[c]["fleiss_kappa"] for c in cats_k]
    colors = ["#d1495b" if k < 0.6 else "#66a182" if k < 0.8 else "#2e4057" for k in kvals]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(cats_k)), kvals, color=colors)
    plt.yticks(range(len(cats_k)), cats_k)
    for thr, lab in ((0.6, "moderate"), (0.8, "substantial")):
        plt.axvline(thr, color="grey", linestyle="--", linewidth=1)
        plt.text(thr, len(cats_k) - 0.5, lab, rotation=90, va="top", fontsize=8, color="grey")
    plt.xlabel("Fleiss' kappa (per category, present/absent)")
    plt.title(f"Per-category multi-label agreement (threshold {threshold})")
    plt.tight_layout()
    _save(plt, report_dir / "category_agreement_kappa.png")

    # (d) Single-label overview, grouped so like is compared with like:
    #     raw agreement (%), chance-corrected agreement (kappa/alpha), and the
    #     per-pair Cohen kappas. Fleiss' kappa and Krippendorff's alpha coincide
    #     for 3 balanced raters, so they share one bar (annotated) rather than
    #     drawing two identical bars.
    fleiss, alpha = single["fleiss_kappa"], single["krippendorff_alpha"]
    kappa_label = "Fleiss' κ ≈ Kripp. α" if abs(fleiss - alpha) < 5e-3 else "Fleiss' κ"
    pair = single["pairwise_cohen_kappa"]
    groups = [
        ("full agreement", single["percent_full_agreement"], "#2e4057"),
        ("mean pairwise agr.", single["mean_pairwise_percent_agreement"], "#2e4057"),
        (kappa_label, fleiss, "#edae49"),
    ] + [(f"Cohen κ: {k}", v, "#66a182") for k, v in pair.items()]
    labels = [g[0] for g in groups]
    values = [g[1] for g in groups]
    colors = [g[2] for g in groups]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), values, color=colors)
    for rect, v in zip(bars, values):
        plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Single-label agreement on primary_category")
    # legend clarifies the three metric families (colour = family).
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(color="#2e4057", label="raw agreement"),
        Patch(color="#edae49", label="chance-corrected (κ / α)"),
        Patch(color="#66a182", label="per-pair Cohen κ"),
    ], loc="upper right")
    plt.tight_layout()
    _save(plt, report_dir / "single_label_agreement.png")

    # (e/f) Comparison of each study model against the reference (primary) LLM.
    ref = result.get("reference_comparison")
    if ref:
        models = list(ref["per_model"])
        metrics = [
            ("primary_agreement", "primary agreement"),
            ("primary_cohen_kappa", "primary Cohen kappa"),
            ("mean_jaccard", "multi-label mean Jaccard"),
            ("mean_per_category_cohen_kappa", "mean per-cat Cohen kappa"),
        ]
        x = range(len(models))
        width = 0.2
        palette = ["#2e4057", "#66a182", "#edae49", "#d1495b"]
        plt.figure(figsize=(12, 6))
        for i, (key, label) in enumerate(metrics):
            vals = [ref["per_model"][m][key] for m in models]
            offs = [xi + (i - (len(metrics) - 1) / 2) * width for xi in x]
            bars = plt.bar(offs, vals, width=width, label=label, color=palette[i])
            for rect, v in zip(bars, vals):
                if not (isinstance(v, float) and math.isnan(v)):
                    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                             f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        plt.xticks(list(x), models, rotation=15, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"Study models vs reference LLM ({ref['reference']}, {ref['n_items']} diffs)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        _save(plt, report_dir / "reference_comparison.png")

        # Heatmap: per-category Cohen kappa vs reference (study models x categories).
        cats = list(next(iter(ref["per_model"].values()))["per_category_cohen_kappa"])
        matrix = [[ref["per_model"][m]["per_category_cohen_kappa"][c] for c in cats] for m in models]
        fig, ax = plt.subplots(figsize=(14, 4 + 0.3 * len(models)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        for r in range(len(models)):
            for cidx in range(len(cats)):
                v = matrix[r][cidx]
                txt = "" if isinstance(v, float) and math.isnan(v) else f"{v:.2f}"
                ax.text(cidx, r, txt, ha="center", va="center", fontsize=7,
                        color="black")
        fig.colorbar(im, ax=ax, label="Cohen kappa vs reference", shrink=0.6)
        ax.set_title(f"Per-category agreement with reference LLM ({ref['reference']})")
        plt.tight_layout()
        _save(plt, report_dir / "reference_category_kappa.png")

    maint = result.get("maintenance_agreement")
    if maint:
        _maintenance_charts(maint, plt, report_dir)


def _maintenance_charts(maint: dict, plt, report_dir: Path) -> None:
    """Render the maintenance-reason agreement figures."""
    # (g) Base type vs parent class: same headline metrics side by side. The
    #     expectation is that the 2-class split agrees more than the 5-type one.
    metrics = [
        ("percent_full_agreement", "full agr."),
        ("mean_pairwise_percent_agreement", "pairwise agr."),
        ("fleiss_kappa", "Fleiss K"),
        ("krippendorff_alpha", "Kripp. alpha"),
        ("mean_pairwise_cohen_kappa", "Cohen K"),
    ]
    base, cls = maint["base_type"], maint["parent_class"]
    x = range(len(metrics))
    width = 0.38
    plt.figure(figsize=(12, 6))
    b1 = plt.bar([xi - width / 2 for xi in x], [base[k] for k, _ in metrics],
                 width=width, label="base type (5)", color="#2e4057")
    b2 = plt.bar([xi + width / 2 for xi in x], [cls[k] for k, _ in metrics],
                 width=width, label="Correction/Enhancement (2)", color="#66a182")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if not (isinstance(h, float) and math.isnan(h)):
                plt.text(rect.get_x() + rect.get_width() / 2, h, f"{h:.2f}",
                         ha="center", va="bottom", fontsize=8)
    plt.xticks(list(x), [lab for _, lab in metrics], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Maintenance-reason agreement: base type vs parent class")
    plt.legend()
    plt.tight_layout()
    _save(plt, report_dir / "maintenance_agreement.png")

    # (h) Per-type ambiguity: presence disagreement (bars) + Fleiss K (line).
    per = maint["per_type"]
    types = sorted(
        per,
        key=lambda t: per[t]["presence_disagreement_rate"]
        if not math.isnan(per[t]["presence_disagreement_rate"]) else -1,
        reverse=True,
    )
    disagree = [per[t]["presence_disagreement_rate"] for t in types]
    kappa = [per[t]["fleiss_kappa"] for t in types]
    y = list(range(len(types)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, disagree, color="#d1495b")
    ax.set_yticks(y)
    ax.set_yticklabels(types)
    ax.invert_yaxis()
    ax.set_xlabel("Presence disagreement rate", color="#d1495b")
    ax.tick_params(axis="x", colors="#d1495b")
    ax.set_xlim(0, max(disagree) * 1.15)
    ax.set_title("Maintenance-type ambiguity across models")

    ax_k = ax.twiny()  # Fleiss' kappa on its own scale (not commensurable with rate)
    ax_k.plot(kappa, y, "o-", color="#1b3a4b")
    ax_k.set_xlabel("Fleiss' kappa (agreement)", color="#1b3a4b")
    ax_k.tick_params(axis="x", colors="#1b3a4b")
    ax_k.set_xlim(0, 1)

    ax.plot([], [], color="#d1495b", linewidth=6, label="presence disagreement rate")
    ax.plot([], [], "o-", color="#1b3a4b", label="Fleiss' kappa")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(plt, report_dir / "maintenance_type_ambiguity.png")

    # (i) Maintenance reason vs the reference LLM, per study model.
    mref = maint.get("reference_comparison")
    if mref:
        models = list(mref["per_model"])
        mm = [
            ("base_agreement", "base agr."),
            ("base_cohen_kappa", "base K"),
            ("class_agreement", "class agr."),
            ("class_cohen_kappa", "class K"),
        ]
        x = range(len(models))
        width = 0.2
        palette = ["#2e4057", "#66a182", "#edae49", "#d1495b"]
        plt.figure(figsize=(12, 6))
        for i, (key, label) in enumerate(mm):
            vals = [mref["per_model"][m][key] for m in models]
            offs = [xi + (i - (len(mm) - 1) / 2) * width for xi in x]
            bars = plt.bar(offs, vals, width=width, label=label, color=palette[i])
            for rect, v in zip(bars, vals):
                if not (isinstance(v, float) and math.isnan(v)):
                    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                             f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        plt.xticks(list(x), models, rotation=15, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"Maintenance reason vs reference LLM ({mref['reference']}, {mref['n_items']} diffs)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        _save(plt, report_dir / "maintenance_reference.png")


def _save(plt, path: Path) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"       {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Root holding <model>/primary_*.jsonl (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Models to treat as raters (default: the three study LLMs).")
    p.add_argument("--reference", type=str, default=DEFAULT_REFERENCE,
                   help=f"Primary/reference LLM each study model is compared against "
                        f"(default: {DEFAULT_REFERENCE}). Use --no-reference to skip.")
    p.add_argument("--no-reference", action="store_true",
                   help="Skip the comparison against the reference LLM.")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Score threshold for the multi-label 'present' set (default: 0.50).")
    p.add_argument("--timestamp", type=str, default=None,
                   help="Pin a single shared run (YYYYMMDD_HHMMSS). Default: merge all runs, latest wins.")
    p.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES_PATH,
                   help=f"categories.json path (default: {DEFAULT_CATEGORIES_PATH}).")
    p.add_argument("--report-dir", type=Path, default=None,
                   help="Where to write outputs (default: <output-dir>/agreement_<now>).")
    p.add_argument("--no-charts", action="store_true",
                   help="Skip rendering the PNG figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    categories = load_category_names(args.categories)
    if not categories:
        sys.exit(f"No categories loaded from {args.categories}; check --categories path.")

    per_model = {m: load_model_records(args.output_dir, m, args.timestamp) for m in args.models}
    for m, recs in per_model.items():
        print(f"[load] {m:24s} -> {len(recs)} records")

    id_sets = [set(recs) for recs in per_model.values()]
    common = set.intersection(*id_sets) if id_sets else set()
    if not common:
        sys.exit("No diff_ids common to all models; nothing to compare.")
    union = set.union(*id_sets)
    print(f"[align] {len(common)} diffs common to all {len(args.models)} models "
          f"({len(union)} in union; {len(union) - len(common)} dropped).")

    # Stable order for reproducibility.
    aligned = {d: [per_model[m][d] for m in args.models] for d in sorted(common)}

    result = analyse(aligned, args.models, categories, args.threshold)

    # Maintenance-reason axis (ISO/IEC/IEEE 14764), when the outputs carry it.
    maint = analyse_maintenance(aligned, args.models)
    if maint is None:
        print("[maintenance] no maintenance block in outputs; skipping maintenance agreement.")
    else:
        result["maintenance_agreement"] = maint

    # Compare each study model against the reference (primary) LLM, on the
    # diffs all study models AND the reference classified.
    if not args.no_reference and args.reference:
        ref_records = load_model_records(args.output_dir, args.reference, None)
        ref_common = sorted(set(aligned) & set(ref_records))
        if not ref_records:
            print(f"[reference] no records for {args.reference}; skipping comparison.")
        elif not ref_common:
            print(f"[reference] no diffs shared with {args.reference}; skipping comparison.")
        else:
            print(f"[reference] {args.reference}: {len(ref_records)} records, "
                  f"{len(ref_common)} shared with study models.")
            ref_aligned = {d: aligned[d] for d in ref_common}
            result["reference_comparison"] = analyse_vs_reference(
                ref_aligned, ref_records, args.models, args.reference, categories, args.threshold
            )
            if maint is not None:
                maint_ref = analyse_maintenance_vs_reference(
                    ref_aligned, ref_records, args.models, args.reference
                )
                if maint_ref is not None:
                    result["maintenance_agreement"]["reference_comparison"] = maint_ref

    print_report(result)

    report_dir = args.report_dir or (
        args.output_dir / f"agreement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    write_outputs(result, report_dir)
    if not args.no_charts:
        make_charts(result, report_dir)


if __name__ == "__main__":
    main()
