import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from pypokerengine.api.game import setup_config, start_poker

def _import(path):
    mod_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)

# run n games of m hands between agent 1 and 2, report chips per game
def run_match(agent1_cls, agent2_cls, name1, name2,num_games,max_round, initial_stack, sb_amount, verbose =0):
    agent1_totals = 0
    agent2_totals = 0
    wins1 = 0
    wins2 = 0
    t0 = time.time()
    for _ in range(num_games):
        config = setup_config(max_round = max_round, initial_stack= initial_stack, small_blind_amount= sb_amount)
        config.register_player(name =name1, algorithm =agent1_cls())
        config.register_player(name= name2, algorithm=agent2_cls())
        result = start_poker(config, verbose = verbose)
        s1 = result["players"][0]["stack"]
        s2 = result["players"][1]["stack"]
        agent1_totals += s1
        agent2_totals += s2
        if s1 > s2:
            wins1 += 1
        elif s2 > s1:
            wins2 += 1
    elapsed = time.time() -t0
    total_hands = num_games * max_round
    chips_per_game_1 = (agent1_totals -num_games*initial_stack) /float(num_games)
    return {
        "games": num_games,
        "hands_total": total_hands,
        "agent1_chips": agent1_totals,
        "agent2_chips": agent2_totals,
        "wins1": wins1,
        "wins2":  wins2,
        "chips_per_game_1": chips_per_game_1,
        "elapsed_sec":  elapsed}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent1", default ="pokeragent.PokerAgent")
    ap.add_argument("--agent2", default ="randomplayer.RandomPlayer")
    ap.add_argument("--name1", default = "A1")
    ap.add_argument("--name2", default="A2")
    ap.add_argument("--games", type=int, default= 20)
    ap.add_argument("--hands", type =int, default= 100)
    ap.add_argument("--stack", type=int, default= 10000)
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