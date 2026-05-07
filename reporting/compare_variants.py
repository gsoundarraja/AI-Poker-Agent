import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reporting.gather_baselines import opponent_suite
from reporting.randomized_opponents import randomized_opponent_specs
from reporting.variants import CheckpointAgent, COMPONENT_VARIANTS, VARIANTS
from training.selfplay import run_match


OUT_ROWS = os.path.join(ROOT, "final_project", "tables", "variant_comparison_rows.csv")
OUT_SUMMARY = os.path.join(ROOT, "final_project", "tables", "variant_comparison_summary.csv")
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=1000)
    ap.add_argument("--small-blind", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--random-per-family", type=int, default=2)
    ap.add_argument("--checkpoint-count", type=int, default=4)
    ap.add_argument("--static-suite", choices=("none", "field", "core", "extended", "all"), default="field")
    ap.add_argument("--include-controls", action="store_true")
    ap.add_argument("--variants", default="")
    ap.add_argument("--seed", type=int, default=683)
    args = ap.parse_args()

    variants = select_variants(args)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows = []
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_ROWS), exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in splits:
            opponents = build_opponents(args, split)
            split_rows = run_split(args, split, variants, opponents, pool)
            rows.extend(split_rows)
            write_rows(OUT_ROWS, rows)
            write_summary(OUT_SUMMARY, summarize(rows))

    write_rows(OUT_ROWS, rows)
    summary = summarize(rows)
    write_summary(OUT_SUMMARY, summary)
    print("wrote {}".format(OUT_ROWS))
    print("wrote {}".format(OUT_SUMMARY))
    print("sec {:.1f}".format(time.time() - t0))
    for row in summary:
        print("{split:10s} {variant:28s} q25={q25_cpg:+.1f} med={median_cpg:+.1f} mean={mean_cpg:+.1f} min={min_cpg:+.1f} win={win_fraction:.3f}".format(**row))


def select_variants(args):
    variants = VARIANTS if args.include_controls else COMPONENT_VARIANTS
    if not args.variants.strip():
        return variants
    requested = {v.strip() for v in args.variants.split(",") if v.strip()}
    selected = [(name, cls) for name, cls in variants if name in requested]
    missing = requested - {name for name, _ in selected}
    if missing:
        raise ValueError("bad variants: {}".format(", ".join(sorted(missing))))
    return selected


def build_opponents(args, split):
    opponents = []
    for name, cls, kwargs in randomized_opponent_specs(
        split=split,
        per_family=args.random_per_family,
        seed=args.seed,
    ):
        opponents.append({
            "name": name,
            "kind": "randomized",
            "cls": cls,
            "kwargs": kwargs,
        })

    if args.static_suite != "none":
        for name, cls in opponent_suite(args.static_suite):
            opponents.append({
                "name": "{}_static_{}".format(split, name),
                "kind": "static",
                "cls": cls,
                "kwargs": {},
            })

    labels = checkpoint_labels(args.checkpoint_count)
    for label in labels:
        strategy = os.path.join(CHECKPOINT_DIR, "cfr_strategy_6336_{}k.json".format(label))
        abstraction = os.path.join(CHECKPOINT_DIR, "cfr_abstraction_6336_{}k.json".format(label))
        if not os.path.exists(strategy) or not os.path.exists(abstraction):
            continue
        opponents.append({
            "name": "{}_checkpoint_{}k".format(split, label),
            "kind": "checkpoint",
            "cls": CheckpointAgent,
            "kwargs": {
                "strategy_path": strategy,
                "abstraction_path": abstraction,
                "use_belief_search": False,
                "use_preflop_lookup": False,
            },
        })
    return opponents


def checkpoint_labels(count):
    labels = available_checkpoint_labels()
    if count <= 0:
        return []
    if count >= len(labels):
        return labels
    if count == 1:
        return [labels[-1]] if labels else []
    step = (len(labels) - 1) / float(count - 1)
    picks = []
    for i in range(count):
        picks.append(labels[int(round(i * step))])
    return list(dict.fromkeys(picks))


def available_checkpoint_labels():
    labels = []
    if not os.path.isdir(CHECKPOINT_DIR):
        return labels
    for name in os.listdir(CHECKPOINT_DIR):
        if not name.startswith("cfr_strategy_6336_") or not name.endswith("k.json"):
            continue
        label = name.rsplit("_", 1)[1][:-6]
        labels.append(label)
    return sorted(labels, key=lambda x: int(x))


def run_split(args, split, variants, opponents, pool):
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, len(variants) * len(opponents))) as tpool:
        futures = {}
        for variant_name, VariantCls in variants:
            for opp in opponents:
                label = "{} {} vs {}".format(split, variant_name, opp["name"])
                print("run {}".format(label))
                futures[(variant_name, opp["name"], opp["kind"])] = tpool.submit(
                    run_match,
                    VariantCls,
                    opp["cls"],
                    variant_name,
                    opp["name"],
                    args.games,
                    args.hands,
                    args.stack,
                    args.small_blind,
                    0,
                    pool,
                    label,
                    agent2_kwargs=opp["kwargs"],
                )
        for (variant_name, opp_name, opp_kind), fut in futures.items():
            res = fut.result()
            row = {
                "split": split,
                "variant": variant_name,
                "opponent": opp_name,
                "opponent_kind": opp_kind,
                "games": res["games"],
                "hands_total": res["hands_total"],
                "agent_chips": res["agent1_chips"],
                "opponent_chips": res["agent2_chips"],
                "agent_wins": res["wins1"],
                "opponent_wins": res["wins2"],
                "chips_per_game": round(res["chips_per_game_1"], 3),
                "best_game_gain": round(res["best_game_gain_1"], 3),
                "worst_game_gain": round(res["worst_game_gain_1"], 3),
                "std_game_gain": round(res["std_game_gain_1"], 3),
                "elapsed_sec": round(res["elapsed_sec"], 3),
            }
            rows.append(row)
            print("{} vs {} {:+.1f} ({}/{})".format(
                variant_name, opp_name, res["chips_per_game_1"], res["wins1"], res["games"]
            ))
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["split"], row["variant"]), []).append(row)
    summary = []
    for (split, variant), vals in grouped.items():
        cpgs = sorted(float(v["chips_per_game"]) for v in vals)
        wins = sum(int(v["agent_wins"]) for v in vals)
        losses = sum(int(v["opponent_wins"]) for v in vals)
        summary.append({
            "split": split,
            "variant": variant,
            "opponents": len(vals),
            "games_per_opponent": vals[0]["games"] if vals else 0,
            "q25_cpg": round(quantile(cpgs, 0.25), 3),
            "median_cpg": round(statistics.median(cpgs) if cpgs else 0.0, 3),
            "mean_cpg": round(sum(cpgs) / float(max(1, len(cpgs))), 3),
            "min_cpg": round(min(cpgs) if cpgs else 0.0, 3),
            "max_cpg": round(max(cpgs) if cpgs else 0.0, 3),
            "win_fraction": round(wins / float(max(1, wins + losses)), 3),
        })
    summary.sort(key=lambda r: (
        r["split"],
        -r["q25_cpg"],
        -r["median_cpg"],
        -r["mean_cpg"],
        -r["min_cpg"],
        -r["win_fraction"],
    ))
    return summary


def quantile(values, q):
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(len(values) - 1, lo + 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def write_rows(path, rows):
    fieldnames = [
        "split", "variant", "opponent", "opponent_kind", "games", "hands_total",
        "agent_chips", "opponent_chips", "agent_wins", "opponent_wins",
        "chips_per_game", "best_game_gain", "worst_game_gain", "std_game_gain",
        "elapsed_sec",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary(path, rows):
    fieldnames = [
        "split", "variant", "opponents", "games_per_opponent", "q25_cpg",
        "median_cpg", "mean_cpg", "min_cpg", "max_cpg", "win_fraction",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    main()
