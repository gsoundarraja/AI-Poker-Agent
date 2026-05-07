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

    def action_distribution(self, valid_actions, hole_card, round_state, player_uuid):
        legal = absn.legal_action_names(valid_actions)
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
        return absn.mask_probs(probs, valid_actions)

    def sample_action(self, probs, rng=None):
        rng = rng or random
        if not probs:
            return "fold"
        probs = absn.normalize_probs(probs)
        pick = rng.random()
        acc = 0.0
        for action in absn.ACTIONS:
            if action not in probs:
                continue
            acc += probs[action]
            if pick <= acc:
                return action
        return list(probs)[-1]

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
        target = absn.parse_infoset_key(key)
        if target is None:
            return None
        best_key = None
        best_dist = None
        for cand in self._keys:
            vals = absn.parse_infoset_key(cand)
            if vals is None or vals[absn.INFOSET_STREET] != target[absn.INFOSET_STREET]:
                continue
            dist = (
                4 * abs(vals[absn.INFOSET_CARD_BUCKET] - target[absn.INFOSET_CARD_BUCKET])
                + 2 * abs(vals[absn.INFOSET_POT_BUCKET] - target[absn.INFOSET_POT_BUCKET])
                + 2 * abs(vals[absn.INFOSET_CALL_BUCKET] - target[absn.INFOSET_CALL_BUCKET])
                + abs(vals[absn.INFOSET_STREET_RAISES] - target[absn.INFOSET_STREET_RAISES])
                + abs(vals[absn.INFOSET_TOTAL_RAISES] - target[absn.INFOSET_TOTAL_RAISES])
                + abs(vals[absn.INFOSET_MAX_BET] - target[absn.INFOSET_MAX_BET])
                + abs(vals[absn.INFOSET_FACING] - target[absn.INFOSET_FACING])
            )
            if best_dist is None or dist < best_dist:
                best_key = cand
                best_dist = dist
        return best_key
