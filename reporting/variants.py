from pokeragent import PokerAgent
from agent import abstraction as _abs
from agent import evaluation as _ev

class NoOppModelAgent(PokerAgent):
    def __init__(self):
        super().__init__()
        self.opp_model.predict_action_distribution = (
            lambda opp_uuid, street, facing_raise=False:
            {"fold": 1.0 / 3, "call": 1.0 / 3, "raise": 1.0 / 3}
        )

class NoDynAbsAgent(PokerAgent):
    def __init__(self):
        super().__init__()

        self._orig_dynamic_bucket_count = _abs.dynamic_bucket_count
        _abs.dynamic_bucket_count = lambda street, pot, stack: 8

    def __del__(self):
        try:
            _abs.dynamic_bucket_count = self._orig_dynamic_bucket_count
        except Exception:
            pass

class NoSearchAgent(PokerAgent):
    def declare_action(self, valid_actions, hole_card, round_state):
        import time
        try:
            self._ensure_uuid(round_state)
            from agent import features as _feat

            feats = _feat.extract_state_features(
                round_state, hole_card, self.uuid, self.initial_stack,
            )
            remaining = 0.30
            win_rate = _ev.adaptive_win_rate(
                hole_card, feats["community"], time_budget=remaining,
            )

            fold_eq = self.opp_model.fold_equity(feats["opp_uuid"], feats["street"])
            scores = _ev.action_scores(feats, win_rate, fold_eq,
                                       self.weights, valid_actions)
            if not scores:
                return self._default(valid_actions)
            action = max(scores, key=lambda k: scores[k])
            return self._ensure_legal(action, valid_actions)
        except Exception:
            return self._default(valid_actions)
        
class NoTuningAgent(PokerAgent):
    def _load_weights(self, path):
        return dict(_ev.DEFAULT_WEIGHTS)
    
class NoPreflopCacheAgent(PokerAgent):
    def __init__(self):
        super().__init__()
        _ev._PREFLOP_HS_CACHE = None
        _ev._PREFLOP_HS_LOAD_ATTEMPTED = True


VARIANTS = [
    ("-OppModel",     NoOppModelAgent),
    ("-DynAbs",       NoDynAbsAgent),
    ("-Search",       NoSearchAgent),
    ("-Tuning",       NoTuningAgent),
    ("-PreflopCache", NoPreflopCacheAgent),
]
