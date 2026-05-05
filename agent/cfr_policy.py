import json
import os
import random

from . import cfr_abstraction as absn
from .preflop_lookup import PreflopLookup


class CFRPolicy:
    def __init__(self, strategy=None, abstraction=None, use_preflop_lookup=False):
        self.strategy = strategy or {}
        self.abstraction = abstraction or absn.default_abstraction()
        self._keys = list(self.strategy.keys())
        self._preflop = (
            PreflopLookup.from_policy(self.strategy, self.abstraction)
            if use_preflop_lookup else None
        )

    @classmethod
    def load(cls, strategy_path, abstraction_path, use_preflop_lookup=False):
        abstraction = absn.load_abstraction(abstraction_path)
        strategy = {}
        if strategy_path and os.path.exists(strategy_path):
            with open(strategy_path, "r") as f:
                payload = json.load(f)
            strategy = payload.get("average_strategy", payload if isinstance(payload, dict) else {})
        return cls(
            strategy=strategy,
            abstraction=abstraction,
            use_preflop_lookup=use_preflop_lookup,
        )

    def choose_action(self, valid_actions, hole_card, round_state, player_uuid, rng=None):
        rng = rng or random
        probs = self.action_distribution(valid_actions, hole_card, round_state, player_uuid)
        return self.sample_action(probs, rng)

    def action_distribution(self, valid_actions, hole_card, round_state, player_uuid):
        legal = self._legal_actions(valid_actions)
        if not legal:
            return {"fold": 1.0}
        if self._preflop is not None:
            preflop = self._preflop.distribution(
                valid_actions, hole_card, round_state, player_uuid
            )
            if preflop is not None:
                return preflop
        key = absn.runtime_infoset(self.abstraction, round_state, hole_card, player_uuid)
        probs = self._lookup_probs(key)
        masked = {a: max(0.0, float(probs.get(a, 0.0))) for a in legal}
        return self._normalize(masked)

    def sample_action(self, probs, rng=None):
        rng = rng or random
        if not probs:
            return "fold"
        masked = {a: max(0.0, float(p)) for a, p in probs.items()}
        total = sum(masked.values())
        if total <= 1e-12:
            masked = {a: 1.0 for a in probs}
            total = float(len(masked))
        pick = rng.random() * total
        acc = 0.0
        for action in absn.ACTIONS:
            if action not in masked:
                continue
            acc += masked[action]
            if pick <= acc:
                return action
        return list(masked)[-1]

    def _lookup_probs(self, key):
        probs = self.strategy.get(key)
        if probs is not None:
            return probs
        nearest = self._nearest_key(key)
        if nearest is not None:
            return self.strategy[nearest]
        return {a: 1.0 / len(absn.ACTIONS) for a in absn.ACTIONS}

    def _nearest_key(self, key):
        if not self._keys:
            return None
        target = _parse_key(key)
        best_key = None
        best_dist = None
        for cand in self._keys:
            vals = _parse_key(cand)
            if vals[0] != target[0]:
                continue
            dist = (
                4 * abs(vals[3] - target[3])
                + 2 * abs(vals[4] - target[4])
                + 2 * abs(vals[5] - target[5])
                + abs(vals[6] - target[6])
                + abs(vals[7] - target[7])
                + abs(vals[8] - target[8])
                + abs(vals[9] - target[9])
            )
            if best_dist is None or dist < best_dist:
                best_key = cand
                best_dist = dist
        return best_key

    def _normalize(self, probs):
        total = sum(max(0.0, float(v)) for v in probs.values())
        if total <= 1e-12:
            n = max(1, len(probs))
            return {a: 1.0 / n for a in probs}
        return {a: max(0.0, float(v)) / total for a, v in probs.items()}

    def _legal_actions(self, valid_actions):
        legal = [a["action"] for a in valid_actions]
        call_amount = None
        for action in valid_actions:
            if action.get("action") == "call":
                call_amount = action.get("amount")
                break
        if isinstance(call_amount, (int, float)) and call_amount <= 0 and "call" in legal:
            legal = [a for a in legal if a != "fold"]
        return legal


def _parse_key(key):
    return tuple(int(x) for x in key.split("|"))
