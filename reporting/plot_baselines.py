import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "final_project", ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(ROOT, "final_project", ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = os.path.join(ROOT, "final_project", "tables", "baselines.csv")
SUMMARY_PATH = os.path.join(ROOT, "final_project", "tables", "baseline_summary.csv")
OUT_PATH = os.path.join(ROOT, "final_project", "figures", "baselines.pdf")

COLOURS = {
    "RandomPlayer": "#4c72b0",
    "RaisedPlayer": "#dd8452",
    "TightPassive": "#55a868",
    "LooseAggr":    "#c44e52",
    "Raise50": "#8172b3",
    "Raise90": "#937860",
    "PreflopAggroSticky": "#da8bc3",
    "CallingStation": "#8c8c8c",
    "PressureFolder": "#64b5cd",
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
                "best":       float(r.get("best_game_gain", 0.0) or 0.0),
                "worst":      float(r.get("worst_game_gain", 0.0) or 0.0),
                "std":        float(r.get("std_game_gain", 0.0) or 0.0),
            })
    return rows


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit("ERROR: {} not found. Run reporting/gather_baselines.py first.".format(CSV_PATH))
    rows = load(CSV_PATH)
    summary = load_summary(SUMMARY_PATH)

    width = max(9, 0.75 * len(rows) + 4.5)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(width, 3.8))
    labels = [r["opponent"] for r in rows]
    cpgs = [r["cpg"] for r in rows]
    winf = [r["wins1"] / max(r["games"], 1) for r in rows]
    worst = [r["worst"] for r in rows]
    colours = [COLOURS.get(lbl, "#888") for lbl in labels]

    bars_a = ax_a.bar(labels, cpgs, color=colours, edgecolor="black", linewidth=0.4)
    ax_a.axhline(0, color="black", linewidth=0.5)
    ax_a.set_ylabel("PokerAgent chips/game")
    ax_a.set_title("(a) Chips/game")
    ax_a.tick_params(axis="x", rotation=35, labelsize=8)
    for bar, v in zip(bars_a, cpgs):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + (50 if v >= 0 else -200),
                  "{:+.0f}".format(v), ha="center", fontsize=8)

    bars_b = ax_b.bar(labels, winf, color=colours, edgecolor="black", linewidth=0.4)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("PokerAgent game-win fraction")
    ax_b.set_title("(b) Game-win fraction")
    ax_b.tick_params(axis="x", rotation=35, labelsize=8)
    for bar, v in zip(bars_b, winf):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.02,
                  "{:.0%}".format(v), ha="center", fontsize=8)
    ax_b.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")

    bars_c = ax_c.bar(labels, worst, color=colours, edgecolor="black", linewidth=0.4)
    ax_c.axhline(0, color="black", linewidth=0.5)
    ax_c.set_ylabel("Worst single-game gain")
    ax_c.set_title("(c) Downside risk")
    ax_c.tick_params(axis="x", rotation=35, labelsize=8)
    for bar, v in zip(bars_c, worst):
        ax_c.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + (45 if v >= 0 else -115),
                  "{:+.0f}".format(v), ha="center", fontsize=7)

    if summary:
        fig.suptitle(
            "Baseline validation: mean {:+.1f}, min opponent {:+.1f}, worst game {:+.0f}".format(
                float(summary.get("mean_chips_per_game", 0.0)),
                float(summary.get("min_opponent_chips_per_game", 0.0)),
                float(summary.get("worst_game_gain", 0.0)),
            ),
            fontsize=10,
            y=1.03,
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("Wrote {}".format(OUT_PATH))


if __name__ == "__main__":
    main()
