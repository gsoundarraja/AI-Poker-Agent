import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pokeragent import PokerAgent
from reporting.gather_baselines import opponent_suite
from reporting.variants import COMPONENT_VARIANTS, VARIANTS
from training.selfplay import run_match


OUT_CSV = os.path.join(ROOT, "final_project", "tables", "ablations.csv")
SUMMARY_CSV = os.path.join(ROOT, "final_project", "tables", "ablation_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=1000)
    ap.add_argument("--small-blind", type=int, default=10)
    ap.add_argument("--suite", choices=("core", "extended", "field", "all"), default="field")
    ap.add_argument("--mode", choices=("opponents", "head-to-head"), default="opponents")
    ap.add_argument("--include-controls", action="store_true")
    ap.add_argument("--variants", default="",
                    help="Comma-separated variant names to run, e.g. CFRBlueprint,PublicBeliefDynamicSearch")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if args.mode == "head-to-head":
        _run_head_to_head(args)
    else:
        _run_against_opponents(args)


def _run_against_opponents(args):
    variants = _select_variants(args)
    opponents = opponent_suite(args.suite)
    t0 = time.time()
    n_workers = os.cpu_count() or 4
    rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        with ThreadPoolExecutor(max_workers=max(1, len(variants) * len(opponents))) as tpool:
            futures = {}
            for variant_name, VariantCls in variants:
                for opponent_name, OpponentCls in opponents:
                    label = "{} vs {}".format(variant_name, opponent_name)
                    print("Submitting {}".format(label))
                    futures[(variant_name, opponent_name)] = tpool.submit(
                        run_match,
                        VariantCls,
                        OpponentCls,
                        variant_name,
                        opponent_name,
                        args.games,
                        args.hands,
                        args.stack,
                        args.small_blind,
                        0,
                        pool,
                        label,
                    )

            for (variant_name, opponent_name), fut in futures.items():
                res = fut.result()
                row = {
                    "variant": variant_name,
                    "opponent": opponent_name,
                    "games": res["games"],
                    "hands_total": res["hands_total"],
                    "agent_chips": res["agent1_chips"],
                    "opponent_chips": res["agent2_chips"],
                    "agent_wins": res["wins1"],
                    "opponent_wins": res["wins2"],
                    "chips_per_game": round(res["chips_per_game_1"], 1),
                    "best_game_gain": round(res["best_game_gain_1"], 1),
                    "worst_game_gain": round(res["worst_game_gain_1"], 1),
                    "std_game_gain": round(res["std_game_gain_1"], 1),
                    "elapsed_sec": round(res["elapsed_sec"], 1),
                }
                rows.append(row)
                print("  {} vs {}: {:+.1f} ({}/{})".format(
                    variant_name,
                    opponent_name,
                    res["chips_per_game_1"],
                    res["wins1"],
                    res["games"],
                ))

    _write_rows(OUT_CSV, rows, [
        "variant", "opponent", "games", "hands_total",
        "agent_chips", "opponent_chips", "agent_wins", "opponent_wins",
        "chips_per_game", "best_game_gain", "worst_game_gain",
        "std_game_gain", "elapsed_sec",
    ])
    _write_summary(rows, args)
    print("Wrote {} and {} (total {:.1f}s)".format(
        OUT_CSV, SUMMARY_CSV, time.time() - t0
    ))


def _run_head_to_head(args):
    variants = _select_variants(args)
    t0 = time.time()
    n_workers = os.cpu_count() or 4
    rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        with ThreadPoolExecutor(max_workers=len(variants)) as tpool:
            futures = {}
            for name, VariantCls in variants:
                print("Submitting PokerAgent vs {}".format(name))
                futures[name] = tpool.submit(
                    run_match,
                    PokerAgent,
                    VariantCls,
                    "PokerAgent",
                    name,
                    args.games,
                    args.hands,
                    args.stack,
                    args.small_blind,
                    0,
                    pool,
                    name,
                )
            for name, fut in futures.items():
                res = fut.result()
                rows.append({
                    "variant": name,
                    "opponent": "PokerAgentHeadToHead",
                    "games": res["games"],
                    "hands_total": res["hands_total"],
                    "agent_chips": res["agent1_chips"],
                    "opponent_chips": res["agent2_chips"],
                    "agent_wins": res["wins1"],
                    "opponent_wins": res["wins2"],
                    "chips_per_game": round(res["chips_per_game_1"], 1),
                    "best_game_gain": round(res["best_game_gain_1"], 1),
                    "worst_game_gain": round(res["worst_game_gain_1"], 1),
                    "std_game_gain": round(res["std_game_gain_1"], 1),
                    "elapsed_sec": round(res["elapsed_sec"], 1),
                })
                print("  PokerAgent vs {}: {:+.1f} ({}/{})".format(
                    name, res["chips_per_game_1"], res["wins1"], res["games"]
                ))

    _write_rows(OUT_CSV, rows, [
        "variant", "opponent", "games", "hands_total",
        "agent_chips", "opponent_chips", "agent_wins", "opponent_wins",
        "chips_per_game", "best_game_gain", "worst_game_gain",
        "std_game_gain", "elapsed_sec",
    ])
    _write_summary(rows, args)
    print("Wrote {} and {} (total {:.1f}s)".format(
        OUT_CSV, SUMMARY_CSV, time.time() - t0
    ))


def _write_rows(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary(rows, args):
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    summary = []
    for variant, vals in by_variant.items():
        cpgs = [float(v["chips_per_game"]) for v in vals]
        wins = sum(int(v["agent_wins"]) for v in vals)
        losses = sum(int(v["opponent_wins"]) for v in vals)
        summary.append({
            "variant": variant,
            "mode": args.mode,
            "suite": args.suite,
            "opponents": len(vals),
            "games_per_opponent": args.games,
            "hands_per_game": args.hands,
            "mean_chips_per_game": round(sum(cpgs) / max(1, len(cpgs)), 1),
            "min_chips_per_game": round(min(cpgs) if cpgs else 0.0, 1),
            "max_chips_per_game": round(max(cpgs) if cpgs else 0.0, 1),
            "win_fraction": round(wins / float(max(1, wins + losses)), 3),
        })

    summary.sort(key=lambda r: (r["mean_chips_per_game"], r["min_chips_per_game"]), reverse=True)
    _write_rows(SUMMARY_CSV, summary, [
        "variant", "mode", "suite", "opponents", "games_per_opponent",
        "hands_per_game", "mean_chips_per_game", "min_chips_per_game",
        "max_chips_per_game", "win_fraction",
    ])


def _select_variants(args):
    variants = VARIANTS if args.include_controls else COMPONENT_VARIANTS
    if not args.variants.strip():
        return variants
    requested = {v.strip() for v in args.variants.split(",") if v.strip()}
    selected = [(name, cls) for name, cls in variants if name in requested]
    found = {name for name, _ in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError("unknown variants: {}".format(", ".join(missing)))
    return selected


if __name__ == "__main__":
    main()
