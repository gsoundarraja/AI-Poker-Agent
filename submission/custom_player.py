import json
import os
import time

from pypokerengine.players import BasePokerPlayer
from agent import evaluation, features, opponent, search

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "data")
WEIGHTS_PATH = os.path.join(DATA_DIR, "eval_weights.json")
#time to decide
DECIDE_BUDGET = 0.40

class CustomPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.weights = self._load_weights(WEIGHTS_PATH)
        self.opp_model = opponent.OpponentModel()
        self.initial_stack = 10000
        self.uuid = None
        self._opp_raised_this_street = False

    def declare_action(self, valid_actions, hole_card, round_state):
            t0 = time.time()
            try:
                self._ensure_uuid(round_state)
                feats = features.extract_state_features(round_state, hole_card, self.uuid, self.initial_stack)
                #calc win rate via MC
                remaining = DECIDE_BUDGET - (time.time() - t0)
                win_rate = evaluation.adaptive_win_rate(hole_card, feats["community"], time_budget= remaining, street = feats["street"], pot = feats["pot"], stack = feats["agent_stack"])
                # predict opp policy
                opp_dist = self.opp_model.predict_action_distribution(feats["opp_uuid"], feats["street"], facing_raise =(feats["to_call"] > 0))
                # expectiminimax https://en.wikipedia.org/wiki/Expectiminimax
                action = search.choose_best_action(feats, win_rate, opp_dist, valid_actions, self.weights)
                #if cant rais do call
                action = self._downgrade_unaffordable_raise(action, feats, valid_actions)
                return self._ensure_legal(action, valid_actions)
            
            except Exception:
                return self._default(valid_actions)

    def receive_game_start_message(self, game_info):
        rule = game_info.get("rule", {}) if isinstance(game_info, dict) else {}
        self.initial_stack = rule.get("initial_stack", self.initial_stack)

    def receive_round_start_message(self, round_count, hole_card, seats):
        # reset per round state
        self._opp_raised_this_street = False
        for seat in seats:
            uuid = seat.get("uuid")
            if uuid is None:
                continue
            pass

    def receive_street_start_message(self, street, round_state):
        self._opp_raised_this_street = False

    def receive_game_update_message(self, new_action, round_state):
        # parse act to opp
        if not isinstance(new_action, dict):
            return
        uuid = new_action.get("player_uuid")
        action = (new_action.get("action") or "").lower()
        if uuid is None or uuid == self.uuid or action not in ("fold", "call", "raise"):
            # note raise to opp
            if uuid == self.uuid and action == "raise":
                pass
            return
        street = round_state.get("street", "preflop")
        facing_raise = self._opp_raised_this_street is False and self._we_raised_last(round_state)
        self.opp_model.update_with_action(uuid, street, action, facing_raise)
        if action == "raise":
            self._opp_raised_this_street = True

    def receive_round_result_message(self, winners, hand_info, round_state):
        self._opp_raised_this_street = False

    def _load_weights(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                w = dict(evaluation.DEFAULT_WEIGHTS)
                w.update(data)
                return w
            except (OSError, json.JSONDecodeError):
                pass
        return dict(evaluation.DEFAULT_WEIGHTS)

    def _ensure_uuid(self, round_state):
        if self.uuid is not None:
            return
        for seat in round_state.get("seats", []):
            if seat.get("hole_card"):
                self.uuid = seat.get("uuid")
                return

    def _we_raised_last(self, round_state):
        #raise
        street = round_state.get("street", "preflop")
        hist = round_state.get("action_histories", {}).get(street, []) or []
        for entry in reversed(hist):
            a = (entry.get("action") or "").upper()
            if a == "RAISE":
                return entry.get("uuid") == self.uuid
            if a in ("CALL", "CHECK", "FOLD"):
                continue
        return False

    def _ensure_legal(self, action, valid_actions):
        legal = {a["action"] for a in valid_actions}
        if action in legal:
            return action
        return self._default(valid_actions)

    def _downgrade_unaffordable_raise(self, action, feats, valid_actions):
        if action != "raise":
            return action
        cost = feats.get("to_call", 0) + feats.get("raise_increment", 0)
        stack = feats.get("agent_stack", 0)
        if stack <= 0 or stack < cost:
            legal = {a["action"] for a in valid_actions}
            if "call" in legal:
                return "call"
            return self._default(valid_actions)
        return action
    
    def _default(self, valid_actions):
        legal = {a["action"] for a in valid_actions}
        if "call" in legal:
            return "call"
        if "fold" in legal:
            return "fold"
        if "raise" in legal:
            return "raise"
        return "fold"

def setup_ai():
    return CustomPlayer()