import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from agent.opponent_model import DEFAULT_PARAMS, normalize_params
from pokeragent import PokerAgent, STRATEGY_PATH
from reporting.gather_baselines import opponent_suite
from training.selfplay import run_match


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "data", "opponent_model_params.json")
OUT_CSV = os.path.join(ROOT, "final_project", "tables", "opponent_model_tuning.csv")


def candidate_grid():
    base = dict(DEFAULT_PARAMS)
    candidates = []
    no_adjust = dict(base)
    no_adjust.update({
        "max_shift": 0.0,
        "min_samples": 999.0,
        "min_multiplier": 1.0,
        "max_multiplier": 1.0,
    })
    candidates.append(no_adjust)

    conservative = dict(base)
    conservative.update({
        "prior": 2.0,
        "min_samples": 12.0,
        "max_shift": 0.12,
        "min_multiplier": 0.80,
        "max_multiplier": 1.20,
        "opp_raise_fold_scale": 0.7,
        "opp_raise_call_scale": 0.7,
        "opp_raise_reraise_scale": 0.35,
        "opp_raise_deterrence": 0.30,
        "opp_raise_call_bonus": 0.05,
        "vs_our_raise_fold_scale": 0.55,
        "vs_our_raise_reraise_deterrence": 0.80,
        "image_raise_surprise_scale": 0.20,
        "image_fold_reduce_fold_scale": 0.25,
        "image_fold_call_scale": 0.15,
    })
    candidates.append(conservative)

    moderate = dict(conservative)
    moderate.update({
        "min_samples": 8.0,
        "max_shift": 0.20,
        "min_multiplier": 0.70,
        "max_multiplier": 1.35,
    })
    candidates.append(moderate)

    respect_aggression = dict(conservative)
    respect_aggression.update({
        "min_samples": 6.0,
        "max_shift": 0.25,
        "min_multiplier": 0.65,
        "max_multiplier": 1.45,
        "opp_raise_fold_scale": -1.0,
        "opp_raise_call_scale": -0.7,
        "opp_raise_reraise_scale": -0.4,
        "opp_raise_deterrence": 0.45,
        "opp_raise_call_bonus": -0.15,
    })
    candidates.append(respect_aggression)

    strong_respect_aggression = dict(respect_aggression)
    strong_respect_aggression.update({
        "min_samples": 4.0,
        "max_shift": 0.35,
        "min_multiplier": 0.55,
        "max_multiplier": 1.60,
        "opp_raise_fold_scale": -1.4,
        "opp_raise_call_scale": -1.0,
        "opp_raise_reraise_scale": -0.6,
        "opp_raise_deterrence": 0.65,
        "opp_raise_call_bonus": -0.25,
    })
    candidates.append(strong_respect_aggression)

    for prior in (0.5, 1.0, 2.0):
        for min_samples in (2.0, 4.0, 8.0):
            for max_shift in (0.30, 0.45, 0.60):
                p = dict(base)
                p.update({
                    "prior": prior,
                    "min_samples": min_samples,
                    "max_shift": max_shift,
                })
                candidates.append(p)

    for scale in (0.75, 1.25):
        p = dict(base)
        p.update({
            "opp_raise_fold_scale": base["opp_raise_fold_scale"] * scale,
            "opp_raise_call_scale": base["opp_raise_call_scale"] * scale,
            "opp_raise_reraise_scale": base["opp_raise_reraise_scale"] * scale,
        })
        candidates.append(p)

    for lo, hi in ((0.60, 1.45), (0.50, 1.60), (0.40, 1.80)):
        p = dict(base)
        p.update({"min_multiplier": lo, "max_multiplier": hi})
        candidates.append(p)

    deduped = []
    seen = set()
    for p in candidates:
        norm = normalize_params(p)
        key = tuple(sorted(norm.items()))
        if key not in seen:
            deduped.append(norm)
            seen.add(key)
    return deduped


def score_candidate(params, opponents, games, hands, stack, small_blind, executor):
    rows = []
    cpgs = []
    wins = 0
    total_games = 0
    for opponent_name, opponent_cls in opponents:
        res = run_match(
            PokerAgent,
            opponent_cls,
            "PokerAgent",
            opponent_name,
            games,
            hands,
            stack,
            small_blind,
            executor=executor,
            agent1_kwargs={"opponent_model_params": params},
        )
        cpg = float(res["chips_per_game_1"])
        cpgs.append(cpg)
        wins += int(res["wins1"])
        total_games += int(res["games"])
        rows.append((opponent_name, cpg, int(res["wins1"]), int(res["games"])))

    mean_cpg = sum(cpgs) / max(1, len(cpgs))
    min_cpg = min(cpgs) if cpgs else 0.0
    score = mean_cpg
    return {
        "score": score,
        "mean_cpg": mean_cpg,
        "min_cpg": min_cpg,
        "win_fraction": wins / float(max(1, total_games)),
        "rows": rows,
    }


def strategy_mtime():
    return os.path.getmtime(STRATEGY_PATH) if os.path.exists(STRATEGY_PATH) else None


def write_selected(path, params, result, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "params": params,
        "selection": {
            "score": result["score"],
            "mean_cpg": result["mean_cpg"],
            "min_cpg": result["min_cpg"],
            "win_fraction": result["win_fraction"],
            "games_per_opponent": args.games,
            "hands_per_game": args.hands,
            "stack": args.stack,
            "small_blind": args.small_blind,
            "final_games_per_opponent": args.final_games,
            "finalists": args.finalists,
            "suite": args.suite,
            "min_weight": args.min_weight,
            "strategy_mtime": strategy_mtime(),
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--final-games", type=int, default=64)
    parser.add_argument("--suite", choices=("core", "extended", "field", "all"), default="field")
    parser.add_argument("--min-weight", type=float, default=0.75)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--allow-strategy-change", action="store_true")
    args = parser.parse_args()

    start_strategy_mtime = strategy_mtime()
    candidates = candidate_grid()
    if args.limit > 0:
        candidates = candidates[:args.limit]
    opponents = opponent_suite(args.suite)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "stage",
        "candidate",
        "score",
        "mean_cpg",
        "min_cpg",
        "win_fraction",
        "params_json",
    ] + [name for name, _ in opponents]

    best_params = None
    best_result = None
    coarse_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, params in enumerate(candidates, 1):
                if (
                    not args.allow_strategy_change
                    and strategy_mtime() != start_strategy_mtime
                ):
                    raise RuntimeError(
                        "cfr_strategy.json changed during tuning. Stop training "
                        "or rerun with --allow-strategy-change if intentional."
                    )
                result = score_candidate(
                    params, opponents, args.games, args.hands, args.stack,
                    args.small_blind, pool
                )
                result["score"] = result["mean_cpg"] + args.min_weight * result["min_cpg"]
                coarse_results.append((result["score"], i, params, result))
                by_opp = {name: cpg for name, cpg, _wins, _games in result["rows"]}
                row = {
                    "stage": "coarse",
                    "candidate": i,
                    "score": round(result["score"], 3),
                    "mean_cpg": round(result["mean_cpg"], 3),
                    "min_cpg": round(result["min_cpg"], 3),
                    "win_fraction": round(result["win_fraction"], 4),
                    "params_json": json.dumps(params, sort_keys=True),
                }
                for name, _ in opponents:
                    row[name] = round(by_opp.get(name, 0.0), 3)
                writer.writerow(row)
                f.flush()
                print(
                    "candidate {}/{} score={:+.1f} mean={:+.1f} min={:+.1f}".format(
                        i, len(candidates), result["score"],
                        result["mean_cpg"], result["min_cpg"],
                    ),
                    flush=True,
                )
                if best_result is None or result["score"] > best_result["score"]:
                    best_result = result
                    best_params = params

            finalists = sorted(coarse_results, reverse=True)[:max(1, args.finalists)]
            print(
                "Rescoring {} finalists with {} games/opponent".format(
                    len(finalists), args.final_games
                ),
                flush=True,
            )
            best_params = None
            best_result = None
            for rank, (_score, candidate_id, params, _old_result) in enumerate(finalists, 1):
                if (
                    not args.allow_strategy_change
                    and strategy_mtime() != start_strategy_mtime
                ):
                    raise RuntimeError(
                        "cfr_strategy.json changed during finalist tuning. Stop training "
                        "or rerun with --allow-strategy-change if intentional."
                    )
                result = score_candidate(
                    params,
                    opponents,
                    args.final_games,
                    args.hands,
                    args.stack,
                    args.small_blind,
                    pool,
                )
                result["score"] = result["mean_cpg"] + args.min_weight * result["min_cpg"]
                by_opp = {name: cpg for name, cpg, _wins, _games in result["rows"]}
                row = {
                    "stage": "final",
                    "candidate": candidate_id,
                    "score": round(result["score"], 3),
                    "mean_cpg": round(result["mean_cpg"], 3),
                    "min_cpg": round(result["min_cpg"], 3),
                    "win_fraction": round(result["win_fraction"], 4),
                    "params_json": json.dumps(params, sort_keys=True),
                }
                for name, _ in opponents:
                    row[name] = round(by_opp.get(name, 0.0), 3)
                writer.writerow(row)
                f.flush()
                print(
                    "finalist {}/{} candidate={} score={:+.1f} mean={:+.1f} min={:+.1f}".format(
                        rank, len(finalists), candidate_id, result["score"],
                        result["mean_cpg"], result["min_cpg"],
                    ),
                    flush=True,
                )
                if best_result is None or result["score"] > best_result["score"]:
                    best_result = result
                    best_params = params

    if not args.no_write:
        write_selected(OUT_JSON, best_params, best_result, args)
    print("Wrote {}".format(OUT_CSV))
    if args.no_write:
        print("Skipped {}".format(OUT_JSON))
    else:
        print("Wrote {}".format(OUT_JSON))
    print("Best score={:+.1f} mean={:+.1f} min={:+.1f}".format(
        best_result["score"], best_result["mean_cpg"], best_result["min_cpg"],
    ))


if __name__ == "__main__":
    main()
