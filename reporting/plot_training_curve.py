import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "training_curve.csv")
OUT_PATH = os.path.join(ROOT, "final_project", "figures", "training_curve.pdf")


def load_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "sweep":    int(r["sweep"]),
                "key":      r["key"],
                "delta":    float(r["delta_chips_per_game"]),
                "accepted": r["accepted"] == "True",
                "value":    float(r["value_after"]),
            })
    return rows


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit("ERROR: {} not found. Run training/tune_weights.py first.".format(CSV_PATH))
    rows = load_rows(CSV_PATH)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 3.8))

    xs = list(range(len(rows)))
    deltas = [r["delta"] for r in rows]
    colors = ["#2ca02c" if r["accepted"] else "#999999" for r in rows]
    ax_a.bar(xs, deltas, color=colors, edgecolor="black", linewidth=0.4)
    ax_a.axhline(0, color="black", linewidth=0.5)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(
        ["{}.{}".format(r["sweep"], r["key"].replace("w_", "").replace("bias_", "b_"))
         for r in rows],
        rotation=70, ha="right", fontsize=7.5,
    )
    ax_a.set_ylabel("Best neighbour improvement (chips/game)")
    ax_a.set_title("(a) Per-perturbation improvement")

    from matplotlib.patches import Patch
    ax_a.legend(handles=[
        Patch(facecolor="#2ca02c", edgecolor="black", label="accepted"),
        Patch(facecolor="#999999", edgecolor="black", label="rejected"),
    ], loc="upper right", fontsize=8)

    keys_seen = sorted({r["key"] for r in rows})

    sys.path.insert(0, ROOT)
    from agent.evaluation import DEFAULT_WEIGHTS
    current = {k: DEFAULT_WEIGHTS.get(k, 0.0) for k in keys_seen}
    traj = {k: [] for k in keys_seen}
    for r in rows:
        current[r["key"]] = r["value"]
        for k in keys_seen:
            traj[k].append(current[k])
    xs = list(range(len(rows)))
    for k in keys_seen:
        ax_b.plot(xs, traj[k], marker="o", markersize=3, linewidth=1.2,
                  label=k.replace("w_", "").replace("bias_", "bias_"))
    ax_b.set_xlabel("Training step (perturbation index)")
    ax_b.set_ylabel("Weight value")
    ax_b.set_title("(b) Weight trajectory")
    ax_b.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax_b.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("Wrote {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
