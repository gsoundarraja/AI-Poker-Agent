import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pypokerengine.players import BasePokerPlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from pokeragent import PokerAgent

from training.selfplay import run_match


OUT_CSV = os.path.join(ROOT, "final_project", "tables", "baselines.csv")


class TightPassive(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return "call"
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


class LooseAggr(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        import random
        self._rng = random.Random(1234)
    def declare_action(self, valid_actions, hole_card, round_state):
        legal = {a["action"] for a in valid_actions}
        r = self._rng.random()
        if r < 0.70 and "raise" in legal:
            return "raise"
        if r < 0.95 and "call" in legal:
            return "call"
        if "fold" in legal:
            return "fold"
        return next(iter(legal))
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


OPPONENTS = [
    ("RandomPlayer",  RandomPlayer),
    ("RaisedPlayer",  RaisedPlayer),
    ("TightPassive",  TightPassive),
    ("LooseAggr",     LooseAggr),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=10000)
    ap.add_argument("--small-blind", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    t0 = time.time()
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "games", "hands_total",
                    "pokeragent_chips", "opponent_chips",
                    "pokeragent_wins", "opponent_wins",
                    "chips_per_game", "elapsed_sec"])
        for name, cls in OPPONENTS:
            print("Running PokerAgent vs {}...".format(name))
            res = run_match(
                PokerAgent, cls, "PokerAgent", name,
                num_games=args.games, max_round=args.hands,
                initial_stack=args.stack, sb_amount=args.small_blind,
                verbose=0,
            )
            w.writerow([name, res["games"], res["hands_total"],
                        res["agent1_chips"], res["agent2_chips"],
                        res["wins1"], res["wins2"],
                        round(res["chips_per_game_1"], 1),
                        round(res["elapsed_sec"], 1)])
            f.flush()
            print("  chips/game: {:+.1f}  ({}/{} games  in {:.1f}s)"
                  .format(res["chips_per_game_1"], res["wins1"], res["games"],
                          res["elapsed_sec"]))
    print("Wrote {}  (total {:.1f}s)".format(OUT_CSV, time.time() - t0))


if __name__ == "__main__":
    main()
