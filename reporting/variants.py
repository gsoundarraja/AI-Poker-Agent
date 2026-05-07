import random

from pypokerengine.players import BasePokerPlayer

from agent import cfr_abstraction as absn
from agent.cfr_policy import CFRPolicy
from pokeragent import PokerAgent


class UniformPolicy:
    def action_distribution(self, valid_actions, hole_card, round_state, player_uuid):
        legal = absn.legal_action_names(valid_actions)
        if not legal:
            return {"fold": 1.0}
        p = 1.0 / len(legal)
        return {action: p for action in legal}

    def sample_action(self, probs, rng=None):
        rng = rng or random
        if not probs:
            return "fold"
        probs = absn.normalize_probs(probs)
        pick = rng.random()
        acc = 0.0
        for action, prob in probs.items():
            acc += prob
            if pick <= acc:
                return action
        return list(probs)[-1]


class NoPolicyAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_belief_search=False, use_preflop_lookup=False)
        self._policy = UniformPolicy()


class CFRAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_belief_search=False, use_preflop_lookup=False)


class PreflopAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_belief_search=False, use_preflop_lookup=True)


class SearchAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_belief_search=True, use_preflop_lookup=False)


class FullAgent(PokerAgent):
    def __init__(self):
        super().__init__(use_belief_search=True, use_preflop_lookup=True)


class CheckpointAgent(PokerAgent):
    def __init__(self, strategy_path, abstraction_path, use_belief_search=False,
                 use_preflop_lookup=False):
        super().__init__(
            use_belief_search=use_belief_search,
            use_preflop_lookup=use_preflop_lookup,
        )
        self._policy = CFRPolicy.load(
            strategy_path,
            abstraction_path,
            use_preflop_lookup=use_preflop_lookup,
        )
        if self._belief_search is not None:
            self._belief_search.policy = self._policy


class _BasePlayer(BasePokerPlayer):
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


class RandomAgent(_BasePlayer):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(683)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = absn.legal_action_names(valid_actions)
        return self._rng.choice(legal or ["fold"])


class CallAgent(_BasePlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        legal = set(absn.legal_action_names(valid_actions))
        if "call" in legal:
            return "call"
        return next(iter(legal))


class RaiseAgent(_BasePlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        legal = set(absn.legal_action_names(valid_actions))
        if "raise" in legal:
            return "raise"
        if "call" in legal:
            return "call"
        return next(iter(legal))


VARIANTS = [
    ("CFR", CFRAgent),
    ("Preflop", PreflopAgent),
    ("Search", SearchAgent),
    ("Full", FullAgent),
    ("NoPolicy", NoPolicyAgent),
    ("Random", RandomAgent),
    ("Call", CallAgent),
    ("Raise", RaiseAgent),
]


COMPONENT_VARIANTS = [
    ("CFR", CFRAgent),
    ("Preflop", PreflopAgent),
    ("Search", SearchAgent),
    ("Full", FullAgent),
]
