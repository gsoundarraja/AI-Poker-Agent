import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.belief_search import DEFAULT_PARAMS
from pokeragent import PokerAgent
from reporting.gather_baselines import opponent_suite
from training.selfplay import run_match


OUT_JSON = os.path.join(ROOT, "data", "belief_search_params.json")
OUT_CSV = os.path.join(ROOT, "final_project", "tables", "belief_search_tuning.csv")
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")

# values to try
PARAM_CHOICES = {
    "max_combos": [24, 32, 40, 56, 72],
    "samples": [24, 36, 48, 72],
    "policy_margin_uncertain": [0.06, 0.08, 0.10, 0.13, 0.16],
    "river_pot_units": [22.0, 28.0, 34.0, 42.0, 52.0],
    "river_to_call_units": [3.0, 4.0, 5.0, 7.0],
    "turn_pot_units": [38.0, 50.0, 64.0, 80.0],
    "turn_to_call_units": [6.0, 8.0, 11.0],
    "action_likelihood_floor": [0.015, 0.025, 0.04, 0.06],
    "credibility_samples": [12, 18, 24, 36],
    "credibility_multiplier_base": [0.60, 0.75, 0.90],
    "credibility_multiplier_scale": [0.20, 0.35, 0.50, 0.70],
    "max_fold_to_raise": [0.70, 0.80, 0.90, 0.95],
    "turn_blend_base": [0.04, 0.08, 0.12, 0.16],
    "river_blend_base": [0.08, 0.12, 0.18, 0.24],
    "uncertain_blend_bonus": [0.0, 0.03, 0.05, 0.08],
    "large_pot_blend_units": [28.0, 36.0, 48.0],
    "large_pot_blend_bonus": [0.0, 0.03, 0.05, 0.08],
    "large_call_blend_units": [4.0, 6.0, 8.0],
    "large_call_blend_bonus": [0.0, 0.03, 0.05, 0.08],
    "max_blend": [0.10, 0.16, 0.22, 0.30, 0.40],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--finalists", type=int, default=5)
    ap.add_argument("--final-games", type=int, default=48)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=1000)
    ap.add_argument("--small-blind", type=int, default=10)
    ap.add_argument("--suite", choices=("core", "extended", "field", "all"), default="field")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=68305)
    ap.add_argument("--write-enabled", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    candidates = [dict(DEFAULT_PARAMS)]
    while len(candidates) < args.candidates:
        candidates.append(sample_params(rng))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    opponents = opponent_suite(args.suite)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for idx, params in enumerate(candidates, 1):
            result = evaluate(params, opponents, args.games, args.hands,
                              args.stack, args.small_blind, pool, idx)
            row = make_row("search", idx, params, result)
            rows.append(row)
            print_result(idx, args.candidates, row)
            write_rows(OUT_CSV, rows)

        finalists = sorted(rows, key=lambda r: (float(r["score"]), float(r["min_cpg"])), reverse=True)
        finalists = finalists[:max(1, args.finalists)]
        for rank, row in enumerate(finalists, 1):
            params = json.loads(row["params_json"])
            result = evaluate(params, opponents, args.final_games, args.hands,
                              args.stack, args.small_blind, pool, "final{}".format(rank))
            final_row = make_row("final", rank, params, result)
            rows.append(final_row)
            print("final {} score={:+.1f} mean={:+.1f} min={:+.1f}".format(
                rank,
                float(final_row["score"]),
                float(final_row["mean_cpg"]),
                float(final_row["min_cpg"]),
            ))
            write_rows(OUT_CSV, rows)

    final_rows = [r for r in rows if r["stage"] == "final"] or rows
    best = max(final_rows, key=lambda r: (float(r["score"]), float(r["min_cpg"])))
    best_params = json.loads(best["params_json"])
    payload = {
        "enabled": bool(args.write_enabled),
        "params": best_params,
        "selection": {
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suite": args.suite,
            "games_per_opponent": args.games,
            "final_games_per_opponent": args.final_games,
            "hands_per_game": args.hands,
            "score": float(best["score"]),
            "mean_cpg": float(best["mean_cpg"]),
            "min_cpg": float(best["min_cpg"]),
            "win_fraction": float(best["win_fraction"]),
            "elapsed_sec": round(time.time() - t0, 1),
            "note": "tuned by selfplay",
        },
    }
    write_json(OUT_JSON, payload)
    checkpoint_name = "belief_search_params_tuned_{}_{}.json".format(
        args.suite, time.strftime("%Y%m%d-%H%M%S")
    )
    write_json(os.path.join(CHECKPOINT_DIR, checkpoint_name), payload)
    print("wrote {}".format(OUT_CSV))
    print("wrote {}".format(OUT_JSON))
    print("best score={:+.1f} mean={:+.1f} min={:+.1f} enabled={}".format(
        float(best["score"]),
        float(best["mean_cpg"]),
        float(best["min_cpg"]),
        payload["enabled"],
    ))


def sample_params(rng):
    params = dict(DEFAULT_PARAMS)
    for key, choices in PARAM_CHOICES.items():
        params[key] = rng.choice(choices)
    return params


def evaluate(params, opponents, games, hands, stack, small_blind, pool, label_id):
    with ThreadPoolExecutor(max_workers=max(1, len(opponents))) as tpool:
        futures = {}
        for opp_name, OppCls in opponents:
            label = "cand{} {}".format(label_id, opp_name)
            futures[opp_name] = tpool.submit(
                run_match,
                PokerAgent,
                OppCls,
                "BeliefSearch",
                opp_name,
                games,
                hands,
                stack,
                small_blind,
                0,
                pool,
                label,
                agent1_kwargs={
                    "use_belief_search": True,
                    "use_preflop_lookup": False,
                    "belief_search_params": params,
                },
            )
        results = {}
        for opp_name, fut in futures.items():
            results[opp_name] = fut.result()
    return results


def make_row(stage, candidate, params, result):
    cpgs = {name: res["chips_per_game_1"] for name, res in result.items()}
    wins = sum(res["wins1"] for res in result.values())
    losses = sum(res["wins2"] for res in result.values())
    mean = sum(cpgs.values()) / float(max(1, len(cpgs)))
    min_cpg = min(cpgs.values()) if cpgs else 0.0
    score = mean
    row = {
        "stage": stage,
        "candidate": candidate,
        "score": round(score, 3),
        "mean_cpg": round(mean, 3),
        "min_cpg": round(min_cpg, 3),
        "win_fraction": round(wins / float(max(1, wins + losses)), 3),
        "params_json": json.dumps(params, sort_keys=True),
    }
    for name, cpg in cpgs.items():
        row[name] = round(cpg, 3)
    return row


def print_result(idx, total, row):
    print("cand {}/{} score={:+.1f} mean={:+.1f} min={:+.1f}".format(
        idx, total, float(row["score"]), float(row["mean_cpg"]), float(row["min_cpg"])
    ))


def write_rows(path, rows):
    fieldnames = ["stage", "candidate", "score", "mean_cpg", "min_cpg", "win_fraction", "params_json"]
    extra = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in extra:
                extra.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames + extra)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
