import random

from pypokerengine.players import BasePokerPlayer

from pokeragent import PokerAgent


class UniformPolicy:
    def action_distribution(self, valid_actions, hole_card, round_state, player_uuid):
        legal = self._legal_actions(valid_actions)
        if not legal:
            return {"fold": 1.0}
        p = 1.0 / len(legal)
        return {action: p for action in legal}

    def sample_action(self, probs, rng=None):
        rng = rng or random
        if not probs:
            return "fold"
        total = sum(max(0.0, float(p)) for p in probs.values())
        if total <= 1e-12:
            return rng.choice(list(probs))
        pick = rng.random() * total
        acc = 0.0
        for action, prob in probs.items():
            acc += max(0.0, float(prob))
            if pick <= acc:
                return action
        return list(probs)[-1]

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


class NoCFRPolicyAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_opponent_model=True)
        self._policy = UniformPolicy()


class CFRNoOpponentModelAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_opponent_model=False)


class UniformLegalAgent(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(683)

    def declare_action(self, valid_actions, hole_card, round_state):
        return self._rng.choice([a["action"] for a in valid_actions])

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


class CallOnlyAgent(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        legal = {a["action"] for a in valid_actions}
        if "call" in legal:
            return "call"
        return next(iter(legal))

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


class RaiseOnlyAgent(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        legal = {a["action"] for a in valid_actions}
        if "raise" in legal:
            return "raise"
        if "call" in legal:
            return "call"
        return next(iter(legal))

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


VARIANTS = [
    ("NoCFRPolicy", NoCFRPolicyAgent),
    ("CFRNoOpponentModel", CFRNoOpponentModelAgent),
    ("UniformLegal", UniformLegalAgent),
    ("CallOnly", CallOnlyAgent),
    ("RaiseOnly", RaiseOnlyAgent),
]
