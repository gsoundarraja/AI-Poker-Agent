import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")
TABLE_DIR = os.path.join(ROOT, "final_project", "tables")
BASELINES_CSV = os.path.join(TABLE_DIR, "baselines.csv")


DEFAULT_CHECKPOINTS = [
    "138000",
    "140004",
    "141918",
    "142002",
    "143004",
]

GROUPS = {
    "core": {"RandomPlayer", "RaisedPlayer", "TightPassive", "LooseAggr"},
    "extended_extra": {"Raise50", "Raise90", "PreflopAggroSticky", "CallingStation", "PressureFolder"},
    "field": {"ValueRegular", "NittyValue", "BalancedPressure", "ShowdownRegular"},
}


def _checkpoint_paths(label):
    stem = "6336_{}k".format(label)
    return {
        "strategy": os.path.join(CHECKPOINT_DIR, "cfr_strategy_{}.json".format(stem)),
        "abstraction": os.path.join(CHECKPOINT_DIR, "cfr_abstraction_{}.json".format(stem)),
        "meta": os.path.join(CHECKPOINT_DIR, "cfr_training_meta_{}.json".format(stem)),
    }


def _restore_checkpoint(label):
    paths = _checkpoint_paths(label)
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("missing checkpoint files: {}".format(", ".join(missing)))
    shutil.copy2(paths["strategy"], os.path.join(ROOT, "data", "cfr_strategy.json"))
    shutil.copy2(paths["abstraction"], os.path.join(ROOT, "data", "cfr_abstraction.json"))
    shutil.copy2(paths["meta"], os.path.join(ROOT, "data", "cfr_training_meta.json"))


def _read_baselines():
    with open(BASELINES_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def _summarize(rows):
    cpgs = [float(r["chips_per_game"]) for r in rows]
    out = {
        "all_mean": _mean(cpgs),
        "worst_opponent_mean": min(cpgs) if cpgs else 0.0,
        "best_opponent_mean": max(cpgs) if cpgs else 0.0,
    }
    for name, opponents in GROUPS.items():
        vals = [float(r["chips_per_game"]) for r in rows if r["opponent"] in opponents]
        out[name + "_mean"] = _mean(vals)
        out[name + "_worst"] = min(vals) if vals else 0.0
    return out


def _write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=1000)
    ap.add_argument("--small-blind", type=int, default=10)
    args = ap.parse_args()

    per_run_rows = []
    per_opponent_rows = []
    run_id = 0
    for label in args.checkpoints:
        _restore_checkpoint(label)
        for rep in range(1, args.reps + 1):
            run_id += 1
            print("checkpoint={} rep={}/{}".format(label, rep, args.reps), flush=True)
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "reporting", "gather_baselines.py"),
                    "--suite", "all",
                    "--games", str(args.games),
                    "--hands", str(args.hands),
                    "--stack", str(args.stack),
                    "--small-blind", str(args.small_blind),
                ],
                cwd=ROOT,
                check=True,
            )
            rows = _read_baselines()
            save_path = os.path.join(
                CHECKPOINT_DIR,
                "baselines_6336_{}k_repeat{:02d}.csv".format(label, rep),
            )
            shutil.copy2(BASELINES_CSV, save_path)

            summary = _summarize(rows)
            summary.update({"checkpoint": label, "rep": rep, "run_id": run_id})
            per_run_rows.append(summary)
            for row in rows:
                copied = dict(row)
                copied.update({"checkpoint": label, "rep": rep, "run_id": run_id})
                per_opponent_rows.append(copied)

    by_checkpoint = defaultdict(list)
    for row in per_run_rows:
        by_checkpoint[row["checkpoint"]].append(row)

    aggregate_rows = []
    metric_names = [
        "all_mean",
        "worst_opponent_mean",
        "best_opponent_mean",
        "core_mean",
        "core_worst",
        "extended_extra_mean",
        "extended_extra_worst",
        "field_mean",
        "field_worst",
    ]
    for label in args.checkpoints:
        runs = by_checkpoint[label]
        out = {"checkpoint": label, "reps": len(runs)}
        for metric in metric_names:
            out[metric] = round(_mean(float(r[metric]) for r in runs), 2)
        aggregate_rows.append(out)

    _write_rows(
        os.path.join(TABLE_DIR, "checkpoint_repeat_runs.csv"),
        ["checkpoint", "rep", "run_id"] + metric_names,
        per_run_rows,
    )
    _write_rows(
        os.path.join(TABLE_DIR, "checkpoint_repeat_opponents.csv"),
        ["checkpoint", "rep", "run_id"] + [k for k in per_opponent_rows[0] if k not in ("checkpoint", "rep", "run_id")],
        per_opponent_rows,
    )
    _write_rows(
        os.path.join(TABLE_DIR, "checkpoint_repeat_summary.csv"),
        ["checkpoint", "reps"] + metric_names,
        aggregate_rows,
    )

    print("\ncheckpoint_repeat_summary")
    for row in aggregate_rows:
        print(row)


if __name__ == "__main__":
    main()
