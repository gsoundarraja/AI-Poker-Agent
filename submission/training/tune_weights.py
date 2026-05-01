# ex python training/tune_weights.py --sweeps 3 --games-per-eval 10
import argparse
import copy
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from agent.evaluation import DEFAULT_WEIGHTS
from training.selfplay import run_match

#weights
TUNABLE_KEYS = ["w_hs", "w_pot_odds","w_position","w_fold_equity", "w_pot_commit","bias_fold", "bias_raise"]
# load weights from dict in mem
def make_agent_class(weights):
    from pokeragent import PokerAgent

    class TunedAgent(PokerAgent):
        def _load_weights(self, path):
            return dict(weights)

    TunedAgent.__name__ = "TunedAgent"
    return TunedAgent

def evaluate_weights(candidate, baseline, games, hands, stack, sb):
    Ca = make_agent_class(candidate)
    Cb = make_agent_class(baseline)
    res = run_match(Ca, Cb, "cand", "base", num_games = games, max_round= hands,initial_stack=stack, sb_amount =sb, verbose=0)
    return res["chips_per_game_1"]

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
    # trianing curve
    curve_path = os.path.join(os.path.dirname(__file__), os.pardir, "data", "training_curve.csv")
    curve_file = open(curve_path, "w", newline="")
    curve = csv.writer(curve_file)
    curve.writerow(["sweep", "key", "delta_chips_per_game", "accepted", "value_after"])

    q_values = {k: 0.0 for k in TUNABLE_KEYS}
    t0 = time.time()
    # for each weight try weight+-delta, test trained agent v baseline  upd Q value, upd baseline for args sweeps
    for sweep in range(args.sweeps):
        print(f"Sweep {sweep+1}/{args.sweeps}")
        for k in TUNABLE_KEYS:
            current = baseline[k]
            delta = max(0.05, abs(current) * args.delta_frac)
            cand_up = copy.deepcopy(baseline)
            cand_up[k] = current + delta
            cand_dn = copy.deepcopy(baseline)
            cand_dn[k] = current- delta
            cpg_up = evaluate_weights(cand_up, baseline,args.games_per_eval, args.hands,args.stack,args.small_blind)
            cpg_dn = evaluate_weights(cand_dn,baseline,args.games_per_eval, args.hands,args.stack, args.small_blind)
            # pick best by cpg
            best_cpg = max(cpg_up, cpg_dn)
            best_cand = cand_up if cpg_up >= cpg_dn else cand_dn
            accepted = best_cpg > 1.0
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