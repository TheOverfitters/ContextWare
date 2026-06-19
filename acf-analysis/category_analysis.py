import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from itertools import combinations


OUTPUT_DIR = "output-report"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    args = parser.parse_args()

    records = load_jsonl(args.input)

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

    print(
        f"\n[+] Analysis exported to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()