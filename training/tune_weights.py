# ex python training/tune_weights.py --sweeps 3 --games-per-eval 10
import argparse
import copy
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from agent.evaluation import DEFAULT_WEIGHTS, set_fast_mc
from training.selfplay import run_match_weighted, run_match_mixed
from raise_player import RaisedPlayer

# less sim for training
set_fast_mc(100)

#weights
TUNABLE_KEYS = ["w_hs", "w_pot_odds","w_position","w_fold_equity", "w_pot_commit","bias_fold", "bias_raise"]

# use multi agent with self play
TRAIN_OPPONENTS = [
    ("self",   0.7, None),
    ("raised", 0.3, RaisedPlayer),
]

def evaluate_weights(candidate, baseline, games, hands, stack, sb, executor, label = None):
    score = 0.0
    for opp_label, w, opp_cls in TRAIN_OPPONENTS:
        sub_label =  f"{label} vs {opp_label}" if label else None
        if opp_cls is None:
            res = run_match_weighted(candidate, baseline, "cand", "base", num_games =games, max_round = hands, initial_stack = stack, sb_amount= sb, executor = executor, label = sub_label)
        else:
            res = run_match_mixed(candidate, opp_cls,"cand", opp_label, num_games = games, max_round = hands, initial_stack = stack, sb_amount = sb, executor = executor, label = sub_label)
        score += w * res["chips_per_game_1"]
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--games-per-eval", type=int, default=8)
    ap.add_argument("--hands", type=int, default =100)
    ap.add_argument("--alpha", type = float, default=0.5)
    ap.add_argument("--delta-frac", type=float, default = 0.25)
    ap.add_argument("--stack", type = int, default= 10000)
    ap.add_argument("--small-blind", type=int, default =10)
    args = ap.parse_args()
    baseline = copy.deepcopy(DEFAULT_WEIGHTS)
    # load prev weights
    w_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "eval_weights.json")
    if os.path.exists(w_path):
        try:
            with open(w_path) as f:
                prev = json.load(f)
            for k in TUNABLE_KEYS:
                if k in prev:
                    baseline[k] = prev[k]
        except (OSError, json.JSONDecodeError):
            pass
    # trianing curve - append if exists, continue sweep numbering from prior runs
    curve_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "training_curve.csv")
    sweep_offset = 0
    file_mode = "w"
    if os.path.exists(curve_path):
        try:
            with open(curve_path) as f:
                rows = list(csv.reader(f))
            if len(rows) > 1:
                sweep_offset = max(int(r[0]) for r in rows[1:]) + 1
                file_mode = "a"
        except (OSError, ValueError, IndexError):
            pass
    curve_file = open(curve_path, file_mode, newline="")
    curve = csv.writer(curve_file)
    if file_mode == "w":
        curve.writerow(["sweep", "key", "delta_chips_per_game", "accepted", "value_after"])

    q_values = {k: 0.0 for k in TUNABLE_KEYS}
    t0 = time.time()
    n_workers = os.cpu_count() or 4
    # pool cand up and dn
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for sweep_idx in range(args.sweeps):
            sweep = sweep_idx + sweep_offset
            print(f"Sweep {sweep+1} (run {sweep_idx+1}/{args.sweeps})")
            for k in TUNABLE_KEYS:
                current = baseline[k]
                delta = max(0.05, abs(current) * args.delta_frac)
                cand_up = copy.deepcopy(baseline)
                cand_up[k] = current + delta
                cand_dn = copy.deepcopy(baseline)
                cand_dn[k] = current- delta
                tag = f"s{sweep+1} {k}"
                with ThreadPoolExecutor(max_workers = 2) as t:
                    f_up = t.submit(evaluate_weights, cand_up, baseline, args.games_per_eval, args.hands, args.stack, args.small_blind, pool, tag + " up")
                    f_dn = t.submit(evaluate_weights, cand_dn, baseline, args.games_per_eval, args.hands, args.stack, args.small_blind, pool, tag + " dn")
                    cpg_up = f_up.result()
                    cpg_dn = f_dn.result()
                # find best cpg
                best_cpg = max(cpg_up, cpg_dn)
                best_cand = cand_up if cpg_up >= cpg_dn else cand_dn
                accepted = best_cpg > 100.0
                if accepted:
                    baseline = best_cand
                    # td update act-value
                    q_values[k] = (1 - args.alpha) * q_values[k] + args.alpha * best_cpg

                print("  {:16s} current={:+.3f}  up={:+.2f}  dn={:+.2f}  accepted={}".format(k, baseline[k], cpg_up, cpg_dn, accepted))
                curve.writerow([sweep, k, best_cpg, accepted, baseline[k]])
                curve_file.flush()
                # write to disk
                with open(w_path, "w") as f:
                    json.dump({k: baseline[k] for k in DEFAULT_WEIGHTS}, f, indent=2)

    curve_file.close()
    print(f"time: {(time.time()-t0):.2f}s, weights: {w_path}")

if __name__ == "__main__":
    main()