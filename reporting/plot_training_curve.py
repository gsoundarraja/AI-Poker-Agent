import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "final_project", ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(ROOT, "final_project", ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = os.path.join(ROOT, "data", "cfr_training_curve.csv")
OUT_PATH = os.path.join(ROOT, "final_project", "figures", "training_curve.pdf")
SUMMARY_PATH = os.path.join(ROOT, "final_project", "tables", "training_curve_summary.csv")


def load(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            traversals = int(float(row.get("total_traversals") or row.get("traversals") or 0))
            elapsed_sec = float(row.get("total_elapsed_sec") or row.get("elapsed_sec") or 0.0)
            rows.append({
                "run_id": row["run_id"],
                "regret_update": row.get("regret_update") or "cfr",
                "traversals": traversals,
                "elapsed_min": elapsed_sec / 60.0,
                "infosets": int(float(row["infosets"] or 0)),
                "utility": float(row["utility_sum_last_batch"] or 0.0),
                "reason": row["checkpoint_reason"],
            })
    return rows


def select_rows(rows, run_id=None, regret_update=None):
    if regret_update:
        rows = [r for r in rows if r["regret_update"] == regret_update]
    if not rows:
        return []
    if run_id is None:
        run_id = rows[-1]["run_id"]
    return [r for r in rows if r["run_id"] == run_id]


def write_summary(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = rows[-1]
    first = rows[0]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "regret_update", "checkpoints",
            "first_traversals", "last_traversals",
            "elapsed_min", "infosets", "last_utility",
        ])
        writer.writerow([
            last["run_id"], last["regret_update"], len(rows),
            first["traversals"], last["traversals"],
            round(last["elapsed_min"], 2), last["infosets"],
            round(last["utility"], 3),
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--regret-update", choices=("cfr", "cfr_plus"))
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        sys.exit("ERROR: {} not found. Run training/train_mccfr.py first.".format(CSV_PATH))
    rows = load(CSV_PATH)
    if not rows:
        sys.exit("ERROR: {} is empty.".format(CSV_PATH))

    rows = select_rows(rows, args.run_id, args.regret_update)
    if not rows:
        sys.exit("ERROR: no rows matched the requested training-curve filters.")
    run_id = rows[-1]["run_id"]
    regret_update = rows[-1]["regret_update"]
    x = [r["traversals"] / 1_000_000.0 for r in rows]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax_a.plot(x, [r["infosets"] for r in rows], marker="o", linewidth=1.4)
    ax_a.set_xlabel("MCCFR traversals (millions)")
    ax_a.set_ylabel("abstract infosets touched")
    ax_a.set_title("(a) Abstraction coverage")
    ax_a.grid(alpha=0.25)

    ax_b.plot(x, [r["utility"] for r in rows], marker="o", linewidth=1.4, color="#c44e52")
    ax_b.axhline(0, color="black", linewidth=0.6)
    ax_b.set_xlabel("MCCFR traversals (millions)")
    ax_b.set_ylabel("sampled utility, last batch")
    ax_b.set_title("(b) Self-play update signal")
    ax_b.grid(alpha=0.25)

    fig.suptitle("MCCFR training progress: {} ({})".format(run_id, regret_update), y=1.03, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    write_summary(SUMMARY_PATH, rows)
    print("Wrote {}".format(args.out))
    print("Wrote {}".format(SUMMARY_PATH))


if __name__ == "__main__":
    main()
