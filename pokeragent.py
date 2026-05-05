import json
import os
import random

from pypokerengine.players import BasePokerPlayer
from agent.cfr_policy import CFRPolicy

try:
    from agent.belief_search import PublicBeliefSearch
except Exception:
    PublicBeliefSearch = None


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "data")
STRATEGY_PATH = os.path.join(DATA_DIR, "cfr_strategy.json")
ABSTRACTION_PATH = os.path.join(DATA_DIR, "cfr_abstraction.json")
BELIEF_SEARCH_PARAMS_PATH = os.path.join(DATA_DIR, "belief_search_params.json")
PREFLOP_LOOKUP_PARAMS_PATH = os.path.join(DATA_DIR, "preflop_lookup_params.json")


class PokerAgent(BasePokerPlayer):
    def __init__(self, use_belief_search=None, use_preflop_lookup=None,
                 belief_search_params=None):
        super().__init__()
        self.uuid = None
        self._rng = random.Random()
        if use_preflop_lookup is None:
            use_preflop_lookup = _load_enabled(PREFLOP_LOOKUP_PARAMS_PATH)
        self._policy = CFRPolicy.load(
            STRATEGY_PATH,
            ABSTRACTION_PATH,
            use_preflop_lookup=use_preflop_lookup,
        )
        self._belief_search = None
        if use_belief_search is None:
            use_belief_search = _load_enabled(BELIEF_SEARCH_PARAMS_PATH)
        if use_belief_search and PublicBeliefSearch is not None:
            params = _load_params(BELIEF_SEARCH_PARAMS_PATH)
            if belief_search_params:
                params.update(belief_search_params)
            self._belief_search = PublicBeliefSearch(
                self._policy,
                rng=self._rng,
                **params,
            )

    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._ensure_uuid(round_state, from_ask=True)
            probs = self._policy.action_distribution(
                valid_actions=valid_actions,
                hole_card=hole_card,
                round_state=round_state,
                player_uuid=self.uuid,
            )
            if self._belief_search is not None:
                probs = self._belief_search.maybe_adjust_distribution(
                    probs, valid_actions, hole_card, round_state, self.uuid
                )
            return self._policy.sample_action(probs, self._rng)
        except Exception:
            return self._uniform_legal_action(valid_actions)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._ensure_uuid_from_seats(seats)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def _ensure_uuid(self, round_state, from_ask=False):
        if self.uuid is not None:
            return
        if from_ask:
            seats = round_state.get("seats", []) or []
            next_player = round_state.get("next_player")
            if isinstance(next_player, int) and 0 <= next_player < len(seats):
                self.uuid = seats[next_player].get("uuid")

    def _ensure_uuid_from_seats(self, seats):
        return

    def _uniform_legal_action(self, valid_actions):
        legal = [a["action"] for a in valid_actions]
        if not legal:
            return "fold"
        return self._rng.choice(legal)


def setup_ai():
    return PokerAgent()


def _load_enabled(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return bool(payload.get("enabled", False))
    except Exception:
        return False


def _load_params(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("params", {})
    except Exception:
        return {}
