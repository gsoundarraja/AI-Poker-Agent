import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from agent import cfr_abstraction as absn
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from pokeragent import PokerAgent

from training.selfplay import run_match


OUT_CSV = os.path.join(ROOT, "final_project", "tables", "baselines.csv")


class _BaseOpponent(BasePokerPlayer):
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


class CallBot(_BaseOpponent):
    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _legal(valid_actions)
        if "call" in legal:
            return "call"
        return next(iter(legal))


class LooseBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(1234)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _legal(valid_actions)
        r = self._rng.random()
        if r < 0.70 and "raise" in legal:
            return "raise"
        if r < 0.95 and "call" in legal:
            return "call"
        if "fold" in legal:
            return "fold"
        return next(iter(legal))


class ProbRaise(_BaseOpponent):
    raise_prob = 0.5
    call_prob = 0.9
    seed = 683

    def __init__(self):
        super().__init__()
        self._rng = random.Random(self.seed)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _legal(valid_actions)
        r = self._rng.random()
        if "raise" in legal and r < self.raise_prob:
            return "raise"
        if "call" in legal and r < self.call_prob:
            return "call"
        if "fold" in legal:
            return "fold"
        return next(iter(legal))


class RaiseHalfBot(ProbRaise):
    raise_prob = 0.50
    call_prob = 0.92
    seed = 1050


class RaiseMostBot(ProbRaise):
    raise_prob = 0.90
    call_prob = 0.98
    seed = 1090


class PreflopRaiseBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6831)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _legal(valid_actions)
        street = round_state.get("street", "preflop")
        r = self._rng.random()
        if street == "preflop" and "raise" in legal and r < 0.85:
            return "raise"
        if street != "preflop" and "raise" in legal and r < 0.25:
            return "raise"
        if "call" in legal:
            return "call"
        return next(iter(legal))


class CallMostlyBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6832)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _legal(valid_actions)
        if "raise" in legal and self._rng.random() < 0.08:
            return "raise"
        if "call" in legal:
            return "call"
        return next(iter(legal))


class FoldToBetBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6833)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _legal(valid_actions)
        facing = _is_facing_bet(round_state, getattr(self, "uuid", None))
        if facing and "fold" in legal and self._rng.random() < 0.70:
            return "fold"
        if "raise" in legal and self._rng.random() < 0.20:
            return "raise"
        if "call" in legal:
            return "call"
        return next(iter(legal))


class ValueBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6840)

    def declare_action(self, valid_actions, hole_card, round_state):
        return _strength_policy(
            valid_actions, hole_card, round_state, getattr(self, "uuid", None),
            self._rng, raise_hi=0.76, call_lo=0.42, fold_lo=0.40, bluff=0.03,
        )


class TightBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6841)

    def declare_action(self, valid_actions, hole_card, round_state):
        return _strength_policy(
            valid_actions, hole_card, round_state, getattr(self, "uuid", None),
            self._rng, raise_hi=0.86, call_lo=0.52, fold_lo=0.56, bluff=0.0,
        )


class PressureBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6842)

    def declare_action(self, valid_actions, hole_card, round_state):
        return _strength_policy(
            valid_actions, hole_card, round_state, getattr(self, "uuid", None),
            self._rng, raise_hi=0.70, call_lo=0.38, fold_lo=0.34, bluff=0.11,
        )


class ShowdownBot(_BaseOpponent):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(6843)

    def declare_action(self, valid_actions, hole_card, round_state):
        return _strength_policy(
            valid_actions, hole_card, round_state, getattr(self, "uuid", None),
            self._rng, raise_hi=0.82, call_lo=0.32, fold_lo=0.28, bluff=0.02,
        )


# validation opponents
CORE_OPPONENTS = [
    ("Random", RandomPlayer),
    ("Raise", RaisedPlayer),
    ("Call", CallBot),
    ("Loose", LooseBot),
]

EXTENDED_OPPONENTS = [
    ("Raise mix", RaiseHalfBot),
    ("Raise often", RaiseMostBot),
    ("Preflop raise", PreflopRaiseBot),
    ("Call mostly", CallMostlyBot),
    ("Fold to bets", FoldToBetBot),
]

FIELD_OPPONENTS = [
    ("Value", ValueBot),
    ("Tight", TightBot),
    ("Pressure", PressureBot),
    ("Showdown", ShowdownBot),
]


def _is_facing_bet(round_state, uuid):
    seats = round_state.get("seats", []) or []
    uuid_to_pos = {seat.get("uuid"): i for i, seat in enumerate(seats)}
    pos = uuid_to_pos.get(uuid)
    if pos is None:
        return False
    street = round_state.get("street", "preflop")
    bets = [0 for _ in seats] or [0, 0]
    for entry in (round_state.get("action_histories", {}) or {}).get(street, []) or []:
        action = (entry.get("action") or "").lower()
        idx = uuid_to_pos.get(entry.get("uuid"))
        if idx is None or action == "fold":
            continue
        if action in ("call", "raise", "smallblind", "bigblind"):
            bets[idx] = max(bets[idx], int(entry.get("amount", 0) or 0))
    return max(bets or [0]) > bets[pos]


def _strength_policy(valid_actions, hole_card, round_state, uuid, rng,
                     raise_hi, call_lo, fold_lo, bluff):
    legal = _legal(valid_actions)
    strength = _hand_strength(hole_card, round_state)
    facing = _is_facing_bet(round_state, uuid)
    street = round_state.get("street", "preflop")
    late = street in ("turn", "river")

    if "raise" in legal:
        raise_cutoff = raise_hi - (0.04 if late else 0.0)
        if strength >= raise_cutoff:
            return "raise"
        if not facing and strength <= 0.30 and rng.random() < bluff:
            return "raise"

    if facing and strength < fold_lo and "fold" in legal:
        return "fold"
    if "call" in legal and (not facing or strength >= call_lo):
        return "call"
    if "fold" in legal:
        return "fold"
    return next(iter(legal))


def _legal(valid_actions):
    return set(absn.legal_action_names(valid_actions))


def _hand_strength(hole_card, round_state):
    board = round_state.get("community_card", []) or []
    if not board:
        try:
            key = absn.preflop_class_from_ids([Card.from_str(c).to_id() for c in hole_card])
            return absn._preflop_class_score(key)
        except Exception:
            return 0.5
    try:
        hole = [Card.from_str(c) for c in hole_card]
        community = [Card.from_str(c) for c in board]
        value = HandEvaluator.eval_hand(hole, community)
    except Exception:
        return 0.5
    category = min(8, max(0, value >> 16))
    hole_hi = max((Card.from_str(c).rank for c in hole_card), default=8)
    street_bonus = 0.02 * max(0, len(board) - 3)
    return min(1.0, 0.18 * category + 0.035 * hole_hi + street_bonus)


def opponent_suite(name, extended=False):
    if extended:
        name = "all"
    if name == "core":
        return CORE_OPPONENTS
    if name == "extended":
        return CORE_OPPONENTS + EXTENDED_OPPONENTS
    if name == "field":
        return FIELD_OPPONENTS
    if name == "all":
        return CORE_OPPONENTS + EXTENDED_OPPONENTS + FIELD_OPPONENTS
    raise ValueError("bad suite: {}".format(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=1000)
    ap.add_argument("--small-blind", type=int, default=10)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--suite", choices=("core", "extended", "field", "all"), default="core")
    args = ap.parse_args()

    opponents = opponent_suite(args.suite, extended=args.extended)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    t0 = time.time()
    n_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        with ThreadPoolExecutor(max_workers = len(opponents)) as tpool:
            futures = {}
            for name, cls in opponents:
                print("run PokerAgent vs {}".format(name))
                futures[name] = (cls, tpool.submit(run_match, PokerAgent, cls, "PokerAgent", name,
                                                   args.games, args.hands, args.stack, args.small_blind, 0, pool, name))
            with open(OUT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["opponent", "games", "hands_total",
                            "pokeragent_chips", "opponent_chips",
                            "pokeragent_wins", "opponent_wins",
                            "chips_per_game", "best_game_gain",
                            "worst_game_gain", "std_game_gain",
                            "elapsed_sec"])
                opponent_cpgs = []
                worst_game_gain = None
                best_game_gain = None
                for name, (_cls, fut) in futures.items():
                    res = fut.result()
                    opponent_cpgs.append(res["chips_per_game_1"])
                    best_game_gain = res["best_game_gain_1"] if best_game_gain is None else max(best_game_gain, res["best_game_gain_1"])
                    worst_game_gain = res["worst_game_gain_1"] if worst_game_gain is None else min(worst_game_gain, res["worst_game_gain_1"])
                    w.writerow([name, res["games"], res["hands_total"],
                                res["agent1_chips"], res["agent2_chips"],
                                res["wins1"], res["wins2"],
                                round(res["chips_per_game_1"], 1),
                                round(res["best_game_gain_1"], 1),
                                round(res["worst_game_gain_1"], 1),
                                round(res["std_game_gain_1"], 1),
                                round(res["elapsed_sec"], 1)])
                    f.flush()
                    print("{} {:+.1f} ({}/{})"
                          .format(name, res["chips_per_game_1"], res["wins1"], res["games"]))
    summary_path = os.path.join(os.path.dirname(OUT_CSV), "baseline_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponents", "games_per_opponent", "hands_per_game",
                    "mean_chips_per_game", "min_opponent_chips_per_game",
                    "max_opponent_chips_per_game", "best_game_gain",
                    "worst_game_gain"])
        w.writerow([
            len(opponents), args.games, args.hands,
            round(sum(opponent_cpgs) / max(1, len(opponent_cpgs)), 1),
            round(min(opponent_cpgs) if opponent_cpgs else 0.0, 1),
            round(max(opponent_cpgs) if opponent_cpgs else 0.0, 1),
            round(best_game_gain if best_game_gain is not None else 0.0, 1),
            round(worst_game_gain if worst_game_gain is not None else 0.0, 1),
        ])
    print("wrote {} ({:.1f}s)".format(OUT_CSV, time.time() - t0))
    print("wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
