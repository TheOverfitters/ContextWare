import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from itertools import combinations

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so `acf_io` (sibling) is importable

# Source diffs, used only to recover each diff's repo (the primary records carry
# diff_id = "<commit_hash>:<filename>" but not the repo).
DEFAULT_DIFFS = SCRIPT_DIR.parent / "outputs" / "git_history_diffs.json"

# Anchor the output to the script's own folder, NOT the current working
# directory: running from the repo root or via the IDE "run" button (whose CWD
# is the workspace root) would otherwise write to a different "output-report"
# and leave the old files untouched -- looking as if nothing was overwritten.
OUTPUT_DIR = str(SCRIPT_DIR / "output-report")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Map diff_id -> repo from the source git_history_diffs.json. Returns {} (so the
# caller degrades gracefully) when the file or helper is unavailable.
def load_repo_map(diffs_json):
    p = Path(diffs_json)
    if not p.exists():
        return {}
    try:
        from acf_io import load_diffs_from_acf_json
        diffs = load_diffs_from_acf_json(p)
    except Exception as exc:
        print(f"[!] Could not read repos from {p.name}: {exc}")
        return {}
    return {
        str(d.get("diff_id", "")): str(d.get("repo", ""))
        for d in diffs
        if d.get("diff_id")
    }

# Reads a JSONL file and returns a list of dictionaries.
def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records

# Calculates primary and general label distributions and the Bias Delta.
def build_distribution(records, threshold):
    categories = set()

    for r in records:
        if "category_scores" not in r:
            print("Available keys:", r.keys())
            raise KeyError("category_scores")

        categories.update(r["category_scores"].keys())

    categories = sorted(categories)

    n_diffs = len(records)

    primary_counts = {c: 0 for c in categories}
    general_counts = {c: 0 for c in categories}

    for r in records:

        scores = r["category_scores"]

        primary = max(scores, key=scores.get)
        primary_counts[primary] += 1

        for cat, score in scores.items():
            if score >= threshold:
                general_counts[cat] += 1

    df = pd.DataFrame(
        {
            "Category": categories,
            "Primary_Count": [primary_counts[c] for c in categories],
            "General_Count": [general_counts[c] for c in categories],
        }
    )

    df["Primary_%"] = (
        df["Primary_Count"] / n_diffs * 100
    )

    df["General_%"] = (
        df["General_Count"] / n_diffs * 100
    )

    df["Bias_Delta_%"] = (
        df["General_%"] - df["Primary_%"]
    )

    return df

# Grouped bar chart comparing Primary vs General label percentages.
def plot_distribution(df, output=None):
    plt.figure(figsize=(14, 7))

    x = range(len(df))

    plt.bar(
        [i - 0.2 for i in x],
        df["Primary_%"],
        width=0.4,
        label="Primary"
    )

    plt.bar(
        [i + 0.2 for i in x],
        df["General_%"],
        width=0.4,
        label="General"
    )

    plt.xticks(
        x,
        df["Category"],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Percentage of diffs")
    plt.title("Primary vs General Label Distribution")
    plt.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        print(f"[+] Graph saved to: {output}")
    else:
        plt.show()

# Dual-axis plot: distribution shape (area) and net change (lollipop).
def plot_distribution_shift(df, output=None):
    plot_df = df.copy()

    plot_df = plot_df.rename(
        columns={
            "Primary_%": "Primary_Pct",
            "General_%": "General_Pct",
            "Bias_Delta_%": "Bias_Pct",
        }
    )

    plot_df = (
        plot_df
        .sort_values("Bias_Pct", ascending=False)
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 14),
        gridspec_kw={"height_ratios": [1.5, 1]}
    )

    ax1.fill_between(
        plot_df.index,
        plot_df["Primary_Pct"],
        alpha=0.3,
        label="Primary Distribution"
    )

    ax1.plot(
        plot_df.index,
        plot_df["Primary_Pct"],
        marker="o",
        linewidth=2,
        label="Primary"
    )

    ax1.plot(
        plot_df.index,
        plot_df["General_Pct"],
        marker="s",
        linewidth=2,
        label="General"
    )

    ax1.set_title(
        "Distribution Shift: Primary vs General",
        fontsize=16,
        fontweight="bold"
    )

    ax1.set_ylabel("Percentage (%)")

    ax1.set_xticks(plot_df.index)

    ax1.set_xticklabels(
        plot_df["Category"],
        rotation=45,
        ha="right"
    )

    ax1.legend()

    colors = [
        "#e74c3c" if x < 0 else "#2ecc71"
        for x in plot_df["Bias_Pct"]
    ]

    ax2.hlines(
        y=plot_df.index,
        xmin=0,
        xmax=plot_df["Bias_Pct"],
        color="grey",
        alpha=0.5
    )

    ax2.scatter(
        plot_df["Bias_Pct"],
        plot_df.index,
        color=colors,
        s=120,
        edgecolors="black",
        zorder=3
    )

    for i, val in enumerate(plot_df["Bias_Pct"]):

        offset = 0.8 if val >= 0 else -0.8

        ax2.text(
            val + offset,
            i,
            f"{val:+.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    ax2.axvline(
        0,
        color="black",
        linewidth=1.5,
        linestyle="--"
    )

    ax2.set_title(
        "Net Change (Bias %) per Category"
    )

    ax2.set_yticks(plot_df.index)

    ax2.set_yticklabels(
        plot_df["Category"]
    )

    ax2.set_xlabel(
        "Percentage Point Change"
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# Heatmap of how often labels appear together in the same diff.
def plot_cooccurrence_matrix(records, threshold=0.5):
    categories = set()

    for r in records:
        categories.update(r["category_scores"].keys())

    categories = sorted(categories)

    idx = {
        cat: i
        for i, cat in enumerate(categories)
    }

    matrix = np.zeros(
        (
            len(categories),
            len(categories)
        )
    )

    for r in records:

        active = [
            cat
            for cat, score in r["category_scores"].items()
            if score >= threshold
        ]

        for a in active:
            matrix[idx[a], idx[a]] += 1

        for a, b in combinations(active, 2):
            matrix[idx[a], idx[b]] += 1
            matrix[idx[b], idx[a]] += 1

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        matrix,
        xticklabels=categories,
        yticklabels=categories,
        cmap="Blues"
    )

    plt.title(
        f"Label Co-occurrence Matrix (threshold={threshold})"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "cooccurrence_matrix.png"
        ),
        dpi=300
    )

    plt.close()

# Heatmap of conditional probability P(B|A) between labels.
def plot_conditional_probability(records, threshold=0.5):
    categories = set()

    for r in records:
        categories.update(r["category_scores"].keys())

    categories = sorted(categories)

    idx = {
        c: i
        for i, c in enumerate(categories)
    }

    occur = np.zeros(len(categories))
    cond = np.zeros(
        (
            len(categories),
            len(categories)
        )
    )

    for r in records:

        active = [
            cat
            for cat, score in r["category_scores"].items()
            if score >= threshold
        ]

        for a in active:

            occur[idx[a]] += 1

            for b in active:
                cond[idx[a], idx[b]] += 1

    for i in range(len(categories)):
        if occur[i] > 0:
            cond[i] /= occur[i]

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        cond,
        xticklabels=categories,
        yticklabels=categories,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.title(
        "Conditional Probability Matrix P(B|A)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "conditional_probability_matrix.png"
        ),
        dpi=300
    )

    plt.close()

# Histogram of the number of activated labels per diff.
def plot_activation_distribution(records, threshold=0.5):
    counts = []

    for r in records:

        active = sum(
            1
            for score in r["category_scores"].values()
            if score >= threshold
        )

        counts.append(active)

    plt.figure(figsize=(10, 6))

    sns.histplot(
        counts,
        bins=max(counts),
        kde=True
    )

    plt.xlabel(
        "Activated Labels"
    )

    plt.ylabel(
        "Number of Diffs"
    )

    plt.title(
        "Activated Labels Per Diff"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "activation_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# Primary vs General percentages sorted by Bias Delta.
def plot_rank_shift(df):
    plot_df = (
        df.copy()
        .sort_values(
            "Bias_Delta_%",
            ascending=False
        )
    )

    plt.figure(figsize=(14, 8))

    x = np.arange(len(plot_df))

    width = 0.4

    plt.bar(
        x - width / 2,
        plot_df["Primary_%"],
        width,
        label="Primary"
    )

    plt.bar(
        x + width / 2,
        plot_df["General_%"],
        width,
        label="General"
    )

    plt.xticks(
        x,
        plot_df["Category"],
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Primary vs General Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "primary_vs_general.png"
        ),
        dpi=300
    )

    plt.close()

# Boxplots of raw scores assigned by the model per category.
def plot_score_distribution(records):
    categories = set()

    for r in records:
        categories.update(r["category_scores"].keys())

    categories = sorted(categories)

    rows = []

    for r in records:

        for cat, score in r["category_scores"].items():
            # Filter out zero scores to avoid skewing the distribution
            if score > 0:
                rows.append(
                    {
                        "Category": cat,
                        "Score": score
                    }
                )

    score_df = pd.DataFrame(rows)

    if score_df.empty:
        print("[!] No non-zero scores found for score distribution plot.")
        return

    plt.figure(figsize=(16, 8))

    sns.boxplot(
        data=score_df,
        x="Category",
        y="Score"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.title(
        "Score Distribution Per Label"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "score_distribution.png"
        ),
        dpi=300
    )

    plt.close()

# Bar chart of processing statuses (ok, recovered, etc.) for pipeline health.
def plot_status_distribution(records):
    statuses = [r.get("status", "else") for r in records]
    status_df = pd.Series(statuses).value_counts().reset_index()
    status_df.columns = ["Status", "Count"]

    plt.figure(figsize=(10, 6))

    color_map = {
        "ok": "#2ecc71",
        "recovered": "#f1c40f",
        "retried_failed": "#e67e22",
        "fallback": "#e74c3c",
        "else": "#95a5a6"
    }

    colors = [color_map.get(s, "#95a5a6") for s in status_df["Status"]]

    sns.barplot(
        data=status_df,
        x="Status",
        y="Count",
        palette=colors,
        hue="Status",
        legend=False
    )

    for i, count in enumerate(status_df["Count"]):
        plt.text(
            i, 
            count + 0.1, 
            str(int(count)), 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )

    plt.title("Processing Status Distribution")
    plt.ylabel("Number of Diffs")
    plt.xlabel("Status")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "status_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Status graph saved to: {output_path}")

# Pie chart showing the proportional distribution of processing statuses.
def plot_status_pie_chart(records):
    statuses = [r.get("status", "else") for r in records]
    status_df = pd.Series(statuses).value_counts().reset_index()
    status_df.columns = ["Status", "Count"]

    plt.figure(figsize=(10, 10))
    
    color_map = {
        "ok": "#2ecc71",
        "recovered": "#f1c40f",
        "retried_failed": "#e67e22",
        "fallback": "#e74c3c",
        "else": "#95a5a6"
    }
    
    colors = [color_map.get(s, "#95a5a6") for s in status_df["Status"]]
    
    plt.pie(
        status_df["Count"], 
        labels=status_df["Status"], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    plt.title("Processing Status Proportion", fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "status_pie_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Status pie chart saved to: {output_path}")

# Exports the bias analysis DataFrame to a CSV, sorted by Bias Delta.
def export_bias_table(df):
    bias_df = (
        df.copy()
        .sort_values(
            "Bias_Delta_%",
            ascending=False
        )
    )

    bias_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "bias_analysis.csv"
        ),
        index=False
    )


# ---------------------------------------------------------------------------
# Maintenance-reason axis (ISO/IEC/IEEE 14764): the "why" of each change.
# These read the per-diff "maintenance" block and are skipped cleanly when the
# input predates that axis. The base type collapses the two adaptive leaves
# ("adaptive (correction)" / "adaptive (enhancement)") back to "adaptive";
# the parent class (Correction / Enhancement) is reported separately.
# ---------------------------------------------------------------------------
MAINT_BASE_ORDER = ["corrective", "preventive", "adaptive", "additive", "perfective"]
MAINT_CLASS_ORDER = ["Correction", "Enhancement"]
MAINT_COLORS = {
    "corrective": "#e74c3c",
    "preventive": "#e67e22",
    "adaptive":   "#3498db",
    "additive":   "#2ecc71",
    "perfective": "#9b59b6",
}
CLASS_COLORS = {"Correction": "#e74c3c", "Enhancement": "#2ecc71"}


def _maint(r):
    m = r.get("maintenance")
    return m if isinstance(m, dict) else {}


# True when at least one record carries a usable maintenance block.
def has_maintenance(records):
    for r in records:
        m = _maint(r)
        if m.get("primary_maintenance_type") or m.get("type_scores"):
            return True
    return False


# Counts of the primary maintenance reason at three granularities:
# leaf (6, adaptive split), base type (5), and parent class (2).
def build_maintenance_distribution(records):
    n = 0
    base_counts, class_counts, leaf_counts = {}, {}, {}

    for r in records:
        m = _maint(r)
        leaf = m.get("primary_maintenance_type")
        if not leaf:
            continue
        n += 1
        base = m.get("maintenance_base_type") or leaf
        cls = m.get("maintenance_class") or ""
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
        base_counts[base] = base_counts.get(base, 0) + 1
        if cls:
            class_counts[cls] = class_counts.get(cls, 0) + 1

    return n, base_counts, class_counts, leaf_counts


def _ordered(keys, preferred):
    return [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]


# Bar chart of the primary maintenance reason (base type) across all diffs.
def plot_maintenance_type_distribution(records):
    n, base_counts, _, _ = build_maintenance_distribution(records)
    if not base_counts:
        print("[!] No maintenance data for type distribution.")
        return

    order = _ordered(base_counts.keys(), MAINT_BASE_ORDER)
    counts = [base_counts[t] for t in order]
    pct = [c / n * 100 for c in counts]
    colors = [MAINT_COLORS.get(t, "#95a5a6") for t in order]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(order, counts, color=colors, edgecolor="black")
    for b, c, p in zip(bars, counts, pct):
        plt.text(
            b.get_x() + b.get_width() / 2, c,
            f"{c}\n({p:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    plt.title("Maintenance Type Distribution (primary reason per diff)")
    plt.ylabel("Number of diffs")
    plt.xlabel("Maintenance type (ISO/IEC/IEEE 14764)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "maintenance_type_distribution.png"), dpi=300)
    plt.close()
    print("[+] Saved maintenance_type_distribution.png")


# Pie of the parent class split (Correction vs Enhancement).
def plot_maintenance_class_distribution(records):
    _, _, class_counts, _ = build_maintenance_distribution(records)
    if not class_counts:
        print("[!] No maintenance class data.")
        return

    order = _ordered(class_counts.keys(), MAINT_CLASS_ORDER)
    counts = [class_counts[c] for c in order]
    colors = [CLASS_COLORS.get(c, "#95a5a6") for c in order]

    plt.figure(figsize=(8, 8))
    plt.pie(
        counts, labels=order, autopct="%1.1f%%", startangle=140,
        colors=colors, wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    plt.title("Correction vs Enhancement (parent class)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "maintenance_class_distribution.png"), dpi=300)
    plt.close()
    print("[+] Saved maintenance_class_distribution.png")


# Boxplots of the raw per-type scores the model assigned (leaf level, non-zero).
def plot_maintenance_score_distribution(records):
    rows = []
    for r in records:
        for t, s in (_maint(r).get("type_scores") or {}).items():
            if isinstance(s, (int, float)) and s > 0:
                rows.append({"Type": t, "Score": s})

    if not rows:
        print("[!] No non-zero maintenance scores.")
        return

    sdf = pd.DataFrame(rows)
    order = sorted(sdf["Type"].unique(), key=lambda t: (t.startswith("adaptive"), t))

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sdf, x="Type", y="Score", order=order)
    plt.xticks(rotation=30, ha="right")
    plt.title("Score Distribution per Maintenance Type (non-zero)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "maintenance_score_distribution.png"), dpi=300)
    plt.close()
    print("[+] Saved maintenance_score_distribution.png")


# Maintenance-reason trajectory across the life of ONE ACF -- the SMALLEST file
# that exercises BOTH classes (>=1 Correction and >=1 Enhancement, >=3 changes),
# so the line actually crosses the two bands and the chart stays compact. A
# SINGLE line traces the change sequence, oscillating
# between two coloured class bands: Correction (red, bottom) and Enhancement
# (green, top). Each change is a dot coloured by its specific type. The parent
# class per change comes from maintenance_class, which also resolves the
# dual-parented "adaptive".
def plot_maintenance_over_time(records, repo_map=None):
    repo_map = repo_map or {}
    CORR = {"corrective", "preventive"}
    ENH = {"additive", "perfective"}

    def class_of(bt: str, cls: str) -> str:
        if cls in ("Correction", "Enhancement"):
            return cls
        if bt in CORR:
            return "Correction"
        if bt in ENH:
            return "Enhancement"
        return ""  # adaptive with no recorded class

    # Group labelled changes per ACF; repo recovered from the diff_id so
    # same-named files in different repos stay distinct. Keep type AND class.
    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for r in records:
        m = _maint(r)
        bt = m.get("maintenance_base_type")
        if not bt:
            continue
        cls = m.get("maintenance_class") or ""
        did = str(r.get("diff_id", ""))
        repo = repo_map.get(did, "") or str(r.get("repo", ""))
        key = (repo, str(r.get("filename", "")))
        groups.setdefault(key, []).append((str(r.get("timestamp", "")) + "|" + did, bt, cls))
    if not groups:
        print("[!] No maintenance data for the lifecycle plot.")
        return

    def ndistinct(items):
        return len({t for _, t, _ in items})

    def classes_of(items):
        return {c for _, bt, cls in items if (c := class_of(bt, cls))}

    # Pick the SMALLEST ACF that exercises BOTH classes (>=1 Correction AND >=1
    # Enhancement) with >=3 changes, so the single line actually oscillates
    # between the two bands. Relax gradually if none qualifies. Tie-break by key.
    both = {"Correction", "Enhancement"}
    eligible = {k: v for k, v in groups.items() if len(v) >= 3 and both <= classes_of(v)}
    if not eligible:
        eligible = {k: v for k, v in groups.items() if len(v) >= 3 and ndistinct(v) >= 2}
    if not eligible:
        eligible = {k: v for k, v in groups.items() if len(v) >= 3}
    if not eligible:
        print("[!] No ACF with >=3 labelled changes for the lifecycle plot.")
        return
    (repo, filename), items = min(sorted(eligible.items()), key=lambda kv: len(kv[1]))
    items.sort(key=lambda t: t[0])
    n = len(items)
    n_types = ndistinct(items)

    xs, ys, dot_colors, types_seen, date_labels = [], [], [], [], []
    for i, (sortkey, bt, cls) in enumerate(items):
        c = class_of(bt, cls)
        ys.append(1.0 if c == "Enhancement" else (0.0 if c == "Correction" else 0.5))
        xs.append(i + 1)
        dot_colors.append(MAINT_COLORS.get(bt, "#95a5a6"))
        types_seen.append(bt)
        # sortkey is "<timestamp>|<diff_id>"; show the commit date (YYYY-MM-DD).
        ts = sortkey.split("|", 1)[0]
        date_labels.append(ts[:10] if ts else f"#{i + 1}")

    fig, ax = plt.subplots(figsize=(min(20.0, max(9.0, n * 0.75)), 4.8))
    # Two class bands.
    ax.axhspan(-0.6, 0.5, color="#e74c3c", alpha=0.12, zorder=0)   # Correction (bottom)
    ax.axhspan(0.5, 1.6, color="#2ecc71", alpha=0.12, zorder=0)    # Enhancement (top)
    ax.axhline(0.5, color="#c8ccd2", linewidth=1.2, zorder=1)

    # Single trajectory line + type-coloured dots.
    ax.plot(xs, ys, color="#555b64", linewidth=2.0, zorder=2)
    ax.scatter(xs, ys, c=dot_colors, s=190, edgecolors="black", linewidths=0.8, zorder=3)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Correction", "Enhancement"], fontweight="bold")
    ax.get_yticklabels()[0].set_color("#c0392b")
    ax.get_yticklabels()[1].set_color("#219653")
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(date_labels, rotation=90, ha="center", fontsize=9)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_xlabel("Commit date  (chronological: first edit → last edit)")
    # Subtle drop line from each dot down to its date tick. Medium grey + dashes
    # so it reads over the coloured class bands without dominating.
    for x, y in zip(xs, ys):
        ax.plot([x, x], [-0.6, y], color="#6b7280", linewidth=1.0,
                linestyle=(0, (4, 3)), alpha=0.65, zorder=1)

    # Legend maps dot colour -> specific maintenance type (only those present).
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=10,
                      markerfacecolor=MAINT_COLORS.get(t, "#95a5a6"),
                      markeredgecolor="black", label=t)
        for t in _ordered(set(types_seen), MAINT_BASE_ORDER)
    ]
    ax.legend(handles=handles, title="Maintenance type (dot colour)",
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

    repo_label = repo or "(repo unknown)"
    file_label = filename or "(unknown file)"
    ax.set_title(
        f"Maintenance across one ACF's life  ·  Correction ↔ Enhancement\n"
        f"{repo_label} — {file_label}  ·  {n} changes, {n_types} types",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "maintenance_lifecycle.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved maintenance_lifecycle.png (ACF: {repo_label}/{file_label}, {n} changes, {n_types} types)")


# Maintenance distribution table (console only) at all three granularities.
def build_maintenance_table(records):
    n, base_counts, class_counts, leaf_counts = build_maintenance_distribution(records)
    if not leaf_counts:
        return None

    rows = []
    for t in _ordered(base_counts.keys(), MAINT_BASE_ORDER):
        rows.append({"Level": "base_type", "Name": t, "Count": base_counts[t], "Pct": base_counts[t] / n * 100})
    for c in _ordered(class_counts.keys(), MAINT_CLASS_ORDER):
        rows.append({"Level": "class", "Name": c, "Count": class_counts[c], "Pct": class_counts[c] / n * 100})
    for lf in sorted(leaf_counts):
        rows.append({"Level": "leaf", "Name": lf, "Count": leaf_counts[lf], "Pct": leaf_counts[lf] / n * 100})

    return pd.DataFrame(rows)


# Runs every maintenance-axis analysis; no-op (with a note) on legacy inputs.
def run_maintenance_analysis(records, repo_map=None):
    if not has_maintenance(records):
        print("\n[i] No 'maintenance' block in input — skipping maintenance analysis.")
        return

    print("\n=== MAINTENANCE-REASON AXIS (ISO/IEC/IEEE 14764) ===\n")
    mdf = build_maintenance_table(records)
    if mdf is not None:
        print(mdf.to_string(index=False))

    plot_maintenance_type_distribution(records)
    plot_maintenance_class_distribution(records)
    plot_maintenance_score_distribution(records)
    plot_maintenance_over_time(records, repo_map)


def main():

    parser = argparse.ArgumentParser(
        description="Bias analysis for multilabel classification outputs"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSONL file"
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Flag threshold (default=0.5)"
    )

    parser.add_argument(
        "--sort",
        choices=[
            "bias",
            "primary",
            "general"
        ],
        default="bias",
        help="Sort criterion"
    )

    parser.add_argument(
        "--csv",
        help="Export results to CSV"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display plot"
    )

    parser.add_argument(
        "--save-plot",
        help="Save plot to PNG"
    )

    parser.add_argument(
        "--diffs-json",
        default=str(DEFAULT_DIFFS),
        help=f"Source diffs, used only to recover each diff's repo for the "
             f"maintenance lifecycle chart (default: {DEFAULT_DIFFS}).",
    )

    args = parser.parse_args()

    records = load_jsonl(args.input)
    repo_map = load_repo_map(args.diffs_json)

    df = build_distribution(
        records,
        args.threshold
    )

    if args.sort == "bias":
        df = df.sort_values(
            "Bias_Delta_%",
            ascending=False
        )

    elif args.sort == "primary":
        df = df.sort_values(
            "Primary_%",
            ascending=False
        )

    elif args.sort == "general":
        df = df.sort_values(
            "General_%",
            ascending=False
        )

    print("\n=== LABEL DISTRIBUTION ===\n")
    print(df.to_string(index=False))

    print("\n=== TOP BIAS LABELS ===\n")
    print(
        df[
            ["Category", "Bias_Delta_%"]
        ].head(10).to_string(index=False)
    )

    if args.csv:
        df.to_csv(
            args.csv,
            index=False
        )
        print(f"\n[+] CSV saved to: {args.csv}")

    if args.plot or args.save_plot:
        plot_distribution(
            df,
            args.save_plot
        )
        
    plot_rank_shift(df)

    plot_distribution_shift(
        df,
        os.path.join(
            OUTPUT_DIR,
            "distribution_shift.png"
        )
    )

    plot_cooccurrence_matrix(
        records,
        args.threshold
    )

    plot_conditional_probability(
        records,
        args.threshold
    )

    plot_activation_distribution(
        records,
        args.threshold
    )

    plot_score_distribution(
        records
    )

    plot_status_distribution(records)
    plot_status_pie_chart(records)

    export_bias_table(df)

    run_maintenance_analysis(records, repo_map)

    print(
        f"\n[+] Analysis exported to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()