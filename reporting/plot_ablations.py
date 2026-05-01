import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "final_project", "tables", "ablations.csv")
OUT_PATH = os.path.join(ROOT, "final_project", "figures", "ablations.pdf")


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit("ERROR: {} not found. Run reporting/run_ablations.py first.".format(CSV_PATH))

    variants, cpgs, win_fracs = [], [], []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            variants.append(r["variant"])
            cpgs.append(float(r["chips_per_game_full"]))
            w1 = int(r["full_wins"]); w2 = int(r["variant_wins"])
            total = max(w1 + w2, 1)
            win_fracs.append(w1 / total)

    order = sorted(range(len(variants)), key=lambda i: cpgs[i], reverse=True)
    variants = [variants[i] for i in order]
    cpgs     = [cpgs[i] for i in order]
    win_fracs = [win_fracs[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.55 * len(variants)))
    colours = ["#2ca02c" if v >= 0 else "#d62728" for v in cpgs]
    bars = ax.barh(variants, cpgs, color=colours, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Full agent chips/game vs. ablated variant  (+ = ablated component helped)")
    ax.set_title("Ablation study: per-component contribution to strength")

    right_pad = max(abs(min(cpgs)), abs(max(cpgs))) * 0.55 + 100
    for bar, v, wf in zip(bars, cpgs, win_fracs):
        sign = "+" if v >= 0 else ""
        label = "{}{:.0f}   ({:.0%} game-wins)".format(sign, v, wf)
        label_x = max(v, 0) + (right_pad * 0.05)
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha="left", fontsize=8)
    ax.set_xlim(min(cpgs) - 100, max(cpgs) + right_pad)
    ax.invert_yaxis()

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("Wrote {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
