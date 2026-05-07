import csv
import io
import os
import zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIG_DIR = os.path.join(ROOT, "final_project", "figures")
TABLE_DIR = os.path.join(ROOT, "final_project", "tables")
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")

VARIANT_LABELS = {
    "CFR": "CFR only",
    "Preflop": "CFR + preflop",
    "Search": "CFR + search",
    "Full": "Full bot",
    "NoPolicy": "No policy",
    "Random": "Random",
    "Call": "Call",
    "Raise": "Raise",
}


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_zip_csv(zip_path, inner_path):
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(inner_path).decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def opponent_group(name):
    simple = {
        "Random", "Raise", "Call", "Loose",
    }
    extended = {
        "Raise mix", "Raise often", "Preflop raise", "Call mostly", "Fold to bets",
    }
    if name in simple:
        return "Basic bots"
    if name in extended:
        return "Extra rule bots"
    return "Poker-like bots"


def variant_label(name):
    return VARIANT_LABELS.get(name, name)


def summarize_baselines(rows):
    cpgs = [float(r["chips_per_game"]) for r in rows]
    groups = defaultdict(list)
    for row in rows:
        groups[opponent_group(row["opponent"])].append(float(row["chips_per_game"]))
    out = {
        "All opponents": mean(cpgs),
        "Worst opponent": min(cpgs) if cpgs else 0.0,
    }
    for group in ("Basic bots", "Extra rule bots", "Poker-like bots"):
        out[group] = mean(groups[group])
    return out


def plot_checkpoint_sweep():
    path = os.path.join(TABLE_DIR, "checkpoint_repeat_summary.csv")
    rows = read_csv(path)
    rows = [
        r for r in rows
        if not (141800 <= int(float(r["checkpoint"])) < 142000)
    ]
    rows = sorted(rows, key=lambda r: float(r["checkpoint"]))
    x = [float(r["checkpoint"]) / 1000.0 for r in rows]
    all_mean = [float(r["all_mean"]) for r in rows]
    worst = [float(r["worst_opponent_mean"]) for r in rows]
    field = [float(r["field_mean"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, all_mean, marker="o", label="Mean over all validation opponents")
    ax.plot(x, field, marker="s", label="Mean over poker-like bots")
    ax.plot(x, worst, marker="^", label="Worst opponent mean")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(142.002, color="#666666", linestyle="--", linewidth=0.9)
    ax.text(142.002, max(all_mean) + 8, "selected", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Training traversals, 6336-infoset CFR model (millions)")
    ax.set_ylabel("Chips/game")
    ax.set_title("Checkpoint selection during MCCFR training")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "checkpoint_sweep.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_training_coverage():
    rows = read_csv(os.path.join(ROOT, "data", "cfr_training_curve.csv"))
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)

    selected = []
    for run_id, vals in by_run.items():
        last = vals[-1]
        infosets = int(float(last["infosets"]))
        regret = last.get("regret_update") or "cfr"
        last_traversals = int(float(last["total_traversals"]))
        first_traversals = int(float(vals[0]["total_traversals"]))
        if infosets in (3104, 6336, 9696) and last_traversals >= 1_000_000:
            selected.append((run_id, regret, infosets, first_traversals, last_traversals, vals))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    colors = {3104: "#4c72b0", 6336: "#55a868", 9696: "#c44e52"}
    for run_id, regret, infosets, first_t, last_t, vals in selected:
        if regret == "cfr_plus":
            label = "9696 infosets, CFR+"
        elif infosets == 6336 and last_t > 100_000_000:
            label = "6336 infosets, CFR"
        elif infosets == 3104:
            label = "3104 infosets, CFR"
        else:
            continue
        x = [int(float(v["total_traversals"])) / 1_000_000.0 for v in vals]
        y = [float(v["utility_sum_last_batch"]) for v in vals]
        ax_a.plot(x, [int(float(v["infosets"])) for v in vals],
                  color=colors.get(infosets, "#888"), alpha=0.65, linewidth=1.1)
        ax_b.plot(x, y, color=colors.get(infosets, "#888"), alpha=0.45, linewidth=0.9, label=label)

    ax_a.set_xlabel("Traversals (millions)")
    ax_a.set_ylabel("Abstract infosets touched")
    ax_a.set_title("(a) Abstraction scale")
    ax_a.grid(alpha=0.25)
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_xlabel("Traversals (millions)")
    ax_b.set_ylabel("Last-batch sampled utility")
    ax_b.set_title("(b) Noisy self-play signal")
    ax_b.grid(alpha=0.25)
    handles, labels = ax_b.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax_b.legend(dedup.values(), dedup.keys(), fontsize=8)
    fig.suptitle("Training runs explored before final checkpoint selection", y=1.03, fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "training_variants.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_cfr_plus_comparison():
    zip_path = os.path.join(CHECKPOINT_DIR, "experiment_cfrplus_9696_141840k_20260504.zip")
    cfr_plus_raw = read_zip_csv(zip_path, "data/checkpoints/baselines_cfrplus_9696_141840k_raw.csv")
    cfr_plus_tuned = read_zip_csv(zip_path, "data/checkpoints/baselines_cfrplus_9696_141840k_tuned.csv")
    selected = read_csv(os.path.join(TABLE_DIR, "baselines.csv"))

    models = [
        ("Selected CFR", summarize_baselines(selected)),
        ("CFR+ raw", summarize_baselines(cfr_plus_raw)),
        ("CFR+ tuned", summarize_baselines(cfr_plus_tuned)),
    ]
    metrics = ["All opponents", "Poker-like bots", "Worst opponent"]
    table_rows = []
    for model, summary in models:
        row = {"model": model}
        for metric in metrics:
            row[metric] = round(summary[metric], 1)
        table_rows.append(row)
    write_csv(
        os.path.join(TABLE_DIR, "cfr_plus_comparison.csv"),
        ["model"] + metrics,
        table_rows,
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.23
    xs = range(len(metrics))
    palette = ["#4c72b0", "#dd8452", "#55a868"]
    for j, (model, summary) in enumerate(models):
        offs = [x + (j - 1) * width for x in xs]
        ax.bar(offs, [summary[m] for m in metrics], width=width,
               label=model, color=palette[j], edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(metrics, rotation=12, ha="right")
    ax.set_ylabel("Chips/game")
    ax.set_title("CFR versus CFR+ experiment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cfr_plus_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_variant_comparison():
    rows = read_csv(os.path.join(TABLE_DIR, "variant_comparison_summary.csv"))
    keep = {"CFR", "Preflop", "Search", "Full"}
    rows = [r for r in rows if r["variant"] in keep]
    order = ["CFR", "Preflop", "Search", "Full"]
    split_rows = {(r["split"], r["variant"]): r for r in rows}
    labels = [variant_label(v) for v in order]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
    for ax, split, title in (
        (ax_a, "validation", "(a) validation split"),
        (ax_b, "test", "(b) held-out test split"),
    ):
        means = [float(split_rows[(split, v)]["mean_cpg"]) for v in order]
        q25 = [float(split_rows[(split, v)]["q25_cpg"]) for v in order]
        y = list(range(len(order)))
        ax.barh(y, means, color="#4c72b0", edgecolor="black", linewidth=0.3, label="mean")
        ax.scatter(q25, y, color="#c44e52", marker="D", label="25th percentile", zorder=3)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("Chips/game")
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    ax_b.legend(fontsize=8, loc="lower right")
    fig.suptitle("Variant comparison on validation and test opponents", y=1.03, fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "variant_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_belief_tuning():
    rows = read_csv(os.path.join(TABLE_DIR, "belief_search_tuning.csv"))
    search = [r for r in rows if r["stage"] == "search"]
    finalists = [r for r in rows if r["stage"] in ("final", "finalist")]
    search_scores = [float(r["score"]) for r in search]
    finalist_scores = [float(r["score"]) for r in finalists]
    finalist_means = [float(r["mean_cpg"]) for r in finalists]
    finalist_mins = [float(r["min_cpg"]) for r in finalists]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax_a.scatter(range(1, len(search_scores) + 1), search_scores,
                 color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax_a.axhline(0, color="black", linewidth=0.6)
    ax_a.set_xlabel("Random candidate")
    ax_a.set_ylabel("Selection score")
    ax_a.set_title("(a) search candidates")
    ax_a.grid(alpha=0.25)

    x = list(range(1, len(finalist_scores) + 1))
    ax_b.bar(x, finalist_scores,
             color="#55a868", edgecolor="black", linewidth=0.3, label="selection score")
    ax_b.plot(x, finalist_means, marker="o", color="#4c72b0", label="mean")
    ax_b.plot(x, finalist_mins, marker="x", color="#c44e52", label="minimum")
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_xlabel("Finalist")
    ax_b.set_ylabel("Chips/game")
    ax_b.set_title("(b) retested finalists")
    ax_b.grid(axis="y", alpha=0.25)
    if finalist_scores:
        best_idx = max(range(len(finalist_scores)), key=lambda i: finalist_scores[i])
        ax_b.annotate(
            "selected",
            xy=(x[best_idx], finalist_means[best_idx]),
            xytext=(36, 0),
            textcoords="offset points",
            arrowprops={"arrowstyle": "<-", "linewidth": 0.7},
            ha="left",
            va="center",
            fontsize=8,
        )
    ax_b.legend(fontsize=7)

    fig.suptitle("Tuning belief search after MCCFR training", y=1.03, fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "belief_search_tuning.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_selected_baselines():
    selected = read_csv(os.path.join(TABLE_DIR, "baselines.csv"))
    if not selected:
        for rep in ("01", "02", "03"):
            path = os.path.join(CHECKPOINT_DIR, "baselines_6336_142002k_repeat{}.csv".format(rep))
            if os.path.exists(path):
                selected.extend(read_csv(path))
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["opponent"]].append(float(row["chips_per_game"]))
    labels = list(grouped)
    means = [mean(v) for v in grouped.values()]
    order = sorted(range(len(labels)), key=lambda i: means[i])
    labels = [labels[i] for i in order]
    means = [means[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    colors = ["#c44e52" if v < 0 else "#55a868" for v in means]
    bars = ax.barh(labels, means, color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean chips/game over repeated validation runs")
    ax.set_title("Current agent against validation baselines")
    for bar, value in zip(bars, means):
        ax.text(value + (18 if value >= 0 else -18),
                bar.get_y() + bar.get_height() / 2,
                "{:+.0f}".format(value),
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "selected_checkpoint_baselines.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_ablation_summary():
    rows = read_csv(os.path.join(TABLE_DIR, "ablation_summary.csv"))
    keep = {"CFR", "Preflop", "Search", "Full"}
    rows = [r for r in rows if r["variant"] in keep]
    order = ["CFR", "Preflop", "Search", "Full"]
    row_by_variant = {r["variant"]: r for r in rows}
    labels = [variant_label(v) for v in order if v in row_by_variant]
    means = [float(row_by_variant[v]["mean_chips_per_game"]) for v in order if v in row_by_variant]
    mins = [float(row_by_variant[v]["min_chips_per_game"]) for v in order if v in row_by_variant]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = list(range(len(labels)))
    ax.barh(y, means, color="#4c72b0", edgecolor="black", linewidth=0.3, label="mean")
    ax.scatter(mins, y, marker="D", color="#c44e52", label="worst opponent", zorder=3)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Chips/game over local validation baselines")
    ax.set_title("Component ablation summary")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ablations.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main():
    ensure_dirs()
    plot_checkpoint_sweep()
    plot_training_coverage()
    plot_cfr_plus_comparison()
    plot_variant_comparison()
    plot_belief_tuning()
    plot_selected_baselines()
    plot_ablation_summary()


if __name__ == "__main__":
    main()
