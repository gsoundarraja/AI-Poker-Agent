import argparse
import importlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from pypokerengine.api.game import setup_config, start_poker

def _import(path):
    mod_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)

# self play partner
def _build_agent(spec):
    kwargs = spec.get("kwargs", {})
    if "weights" in spec:
        from pokeragent import PokerAgent
        class TunedAgent(PokerAgent):
            def _load_weights(self, path):
                return dict(spec["weights"])
        return TunedAgent()
    return spec["cls"](**kwargs)

# setup for processpoool
def _play_one(spec1, spec2, name1, name2, max_round, initial_stack, sb_amount):
    config = setup_config(max_round = max_round, initial_stack = initial_stack, small_blind_amount = sb_amount)
    config.register_player(name = name1, algorithm = _build_agent(spec1))
    config.register_player(name = name2, algorithm = _build_agent(spec2))
    result = start_poker(config, verbose = 0)
    return result["players"][0]["stack"], result["players"][1]["stack"]

def _run_inner(spec1, spec2, name1, name2, num_games, max_round, initial_stack, sb_amount, executor = None, label = None):
    t0 = time.time()
    own = executor is None
    if own:
        workers = min(num_games, os.cpu_count() or 1)
        executor = ProcessPoolExecutor(max_workers = workers)
    try:
        futures = [executor.submit(_play_one, spec1, spec2, name1, name2, max_round, initial_stack, sb_amount) for _ in range(num_games)]
        results = []
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            if label:
                print(f"    [{label}] {i}/{num_games}", flush = True)
    finally:
        if own:
            executor.shutdown()
    a1 = sum(r[0] for r in results)
    a2 = sum(r[1] for r in results)
    gains1 = [s1 - initial_stack for s1, _s2 in results]
    w1 = sum(1 for s1, s2 in results if s1 > s2)
    w2 = sum(1 for s1, s2 in results if s2 > s1)
    mean_gain = sum(gains1) / float(max(1, len(gains1)))
    variance = sum((g - mean_gain) ** 2 for g in gains1) / float(max(1, len(gains1)))
    return {
        "games": num_games,
        "hands_total": num_games * max_round,
        "agent1_chips": a1,
        "agent2_chips": a2,
        "wins1": w1,
        "wins2": w2,
        "chips_per_game_1": mean_gain,
        "best_game_gain_1": max(gains1) if gains1 else 0,
        "worst_game_gain_1": min(gains1) if gains1 else 0,
        "std_game_gain_1": math.sqrt(variance),
        "elapsed_sec": time.time() - t0,
    }

#  n games of m hands, cpg
def run_match(agent1_cls, agent2_cls, name1, name2, num_games, max_round,
              initial_stack, sb_amount, verbose = 0, executor = None, label = None,
              agent1_kwargs = None, agent2_kwargs = None):
    return _run_inner(
        {"cls": agent1_cls, "kwargs": agent1_kwargs or {}},
        {"cls": agent2_cls, "kwargs": agent2_kwargs or {}},
        name1, name2, num_games, max_round, initial_stack, sb_amount,
        executor = executor, label = label,
    )

def run_match_weighted(weights1, weights2, name1, name2, num_games, max_round, initial_stack, sb_amount, executor = None, label = None):
    return _run_inner({"weights": weights1}, {"weights": weights2}, name1, name2, num_games, max_round, initial_stack, sb_amount, executor = executor, label = label)

def run_match_mixed(weights1, opp_cls, name1, name2, num_games, max_round, initial_stack, sb_amount, executor = None, label = None):
    return _run_inner({"weights": weights1}, {"cls": opp_cls}, name1, name2, num_games, max_round, initial_stack, sb_amount, executor = executor, label = label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent1", default ="pokeragent.PokerAgent")
    ap.add_argument("--agent2", default ="randomplayer.RandomPlayer")
    ap.add_argument("--name1", default = "A1")
    ap.add_argument("--name2", default="A2")
    ap.add_argument("--games", type=int, default= 20)
    ap.add_argument("--hands", type =int, default= 100)
    ap.add_argument("--stack", type=int, default= 1000)
    ap.add_argument("--small-blind", type = int, default=10)
    ap.add_argument("--verbose", type=int, default = 0)
    args = ap.parse_args()
    A1 = _import(args.agent1)
    A2 = _import(args.agent2)
    result = run_match(A1, A2, args.name1, args.name2, num_games =args.games, max_round= args.hands,initial_stack = args.stack, sb_amount=args.small_blind, verbose =args.verbose)

    print("Match: {} vs {}".format(args.agent1, args.agent2))
    print("Games: {}".format(result["games"]))
    print("Hands total: {}".format(result["hands_total"]))
    print("{:12s}: {:8d} chips, {} game-wins".format(args.name1, result["agent1_chips"], result["wins1"]))
    print("{:12s}: {:8d} chips, {} game-wins".format(args.name2, result["agent2_chips"], result["wins2"]))
    print("chips/game ({}): {:+.2f}".format(args.name1, result["chips_per_game_1"]))
    print("Elapsed: {:.2}s".format(result["elapsed_sec"]))

if __name__ == "__main__":
    main()
