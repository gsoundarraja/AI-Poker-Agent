import json
import os
import random

from pypokerengine.players import BasePokerPlayer
from agent.cfr_policy import CFRPolicy
from agent.opponent_model import OnlineOpponentModel


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "data")
STRATEGY_PATH = os.path.join(DATA_DIR, "cfr_strategy.json")
ABSTRACTION_PATH = os.path.join(DATA_DIR, "cfr_abstraction.json")
OPPONENT_PARAMS_PATH = os.path.join(DATA_DIR, "opponent_model_params.json")


class CustomPlayer(BasePokerPlayer):
    def __init__(self, use_opponent_model=True, opponent_model_params=None):
        super().__init__()
        self.uuid = None
        self._rng = random.Random()
        self._policy = CFRPolicy.load(STRATEGY_PATH, ABSTRACTION_PATH)
        self._use_opponent_model = use_opponent_model
        params = opponent_model_params
        if params is None:
            params = _load_opponent_model_params(OPPONENT_PARAMS_PATH)
        self._opponent_model = OnlineOpponentModel(**params) if use_opponent_model else None

    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            self._ensure_uuid(round_state, from_ask=True)
            probs = self._policy.action_distribution(
                valid_actions=valid_actions,
                hole_card=hole_card,
                round_state=round_state,
                player_uuid=self.uuid,
            )
            if self._opponent_model is not None:
                probs = self._opponent_model.adjust_distribution(
                    probs, round_state, self.uuid
                )
            return self._policy.sample_action(probs, self._rng)
        except Exception:
            return self._uniform_legal_action(valid_actions)

    def receive_game_start_message(self, game_info):
        if self._opponent_model is not None:
            self._opponent_model.reset()

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._ensure_uuid_from_seats(seats)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        if self._opponent_model is not None:
            self._ensure_uuid(round_state)
            self._opponent_model.observe_action(new_action, round_state, self.uuid)

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
    return CustomPlayer()


def _load_opponent_model_params(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload.get("params", payload)
    except Exception:
        return {}
