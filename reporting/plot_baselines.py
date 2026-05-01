import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "final_project", "tables", "baselines.csv")
OUT_PATH = os.path.join(ROOT, "final_project", "figures", "baselines.pdf")

COLOURS = {
    "RandomPlayer": "#4c72b0",
    "RaisedPlayer": "#dd8452",
    "TightPassive": "#55a868",
    "LooseAggr":    "#c44e52",
}


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "opponent":   r["opponent"],
                "games":      int(r["games"]),
                "cpg":        float(r["chips_per_game"]),
                "wins1":      int(r["pokeragent_wins"]),
            })
    return rows


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit("ERROR: {} not found. Run reporting/gather_baselines.py first.".format(CSV_PATH))
    rows = load(CSV_PATH)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = [r["opponent"] for r in rows]
    cpgs = [r["cpg"] for r in rows]
    winf = [r["wins1"] / max(r["games"], 1) for r in rows]
    colours = [COLOURS.get(lbl, "#888") for lbl in labels]

    bars_a = ax_a.bar(labels, cpgs, color=colours, edgecolor="black", linewidth=0.4)
    ax_a.axhline(0, color="black", linewidth=0.5)
    ax_a.set_ylabel("PokerAgent chips/game")
    ax_a.set_title("(a) Win-rate vs. baseline opponents")
    ax_a.tick_params(axis="x", rotation=25)
    for bar, v in zip(bars_a, cpgs):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + (50 if v >= 0 else -200),
                  "{:+.0f}".format(v), ha="center", fontsize=8)

    bars_b = ax_b.bar(labels, winf, color=colours, edgecolor="black", linewidth=0.4)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("PokerAgent game-win fraction")
    ax_b.set_title("(b) Match-level dominance")
    ax_b.tick_params(axis="x", rotation=25)
    for bar, v in zip(bars_b, winf):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.02,
                  "{:.0%}".format(v), ha="center", fontsize=8)
    ax_b.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("Wrote {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
