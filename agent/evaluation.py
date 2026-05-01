import os
import pickle

from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
from . import abstraction

# less mc eval for training
_FAST_MC_SIMS = int(os.environ["POKERAGENT_FAST_MC"]) if os.environ.get("POKERAGENT_FAST_MC", "").isdigit() else None

def set_fast_mc(n):
    global _FAST_MC_SIMS
    _FAST_MC_SIMS = n
    if n is None:
        os.environ.pop("POKERAGENT_FAST_MC", None)
    else:
        os.environ["POKERAGENT_FAST_MC"] = str(n)

# load precomputed preflop handstrengths
_PREFLOP_HS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "preflop_hs.pkl")
_PREFLOP_HS_CACHE = None

# precomp handstrength lookup from pickle dict
def _preflop_hs(hole_card):
    global _PREFLOP_HS_CACHE
    if _PREFLOP_HS_CACHE is None:
        if os.path.exists(_PREFLOP_HS_PATH):
            with open(_PREFLOP_HS_PATH, "rb") as f:
                _PREFLOP_HS_CACHE = pickle.load(f)
    if not _PREFLOP_HS_CACHE:
        return None
    return _PREFLOP_HS_CACHE.get(abstraction.preflop(hole_card))

# f1 = hand_strength est
# f2 = pots_ods_pressure odds of winning pot
# f3 = position +1 in posiition, -1 out of position (+1 if after, -1 if first)
# f4 = opponent_fold_equity Pr(opp folds|raise)
# f5 = pot_commitment how much of stack already in pot
# default before training
DEFAULT_WEIGHTS = {
    "w_hs": 1.00,
    "w_pot_odds": 0.60,
    "w_position": 0.10,
    "w_fold_equity": 0.35,
    "w_pot_commit": 0.20,
    "bias_fold": -0.50,
    "bias_call": 0.00,
    "bias_raise": -0.10,
}

# MC est of winrate
def win_rate_mc(hole_card, community_card, nb_simulation= 200, nb_player =2):
    #use cache
    if not hole_card:
        return 0.5
    if not community_card:
        cached = _preflop_hs(hole_card)
        if cached is not None:
            return cached
    return estimate_hole_card_win_rate(nb_simulation =nb_simulation, nb_player= nb_player, hole_card = gen_cards(hole_card), community_card = gen_cards(community_card) if community_card else [])

# more MC sims w big budget
def adaptive_win_rate(hole_card, community_card, time_budget, street=None, pot=None, stack=None):
    if _FAST_MC_SIMS is not None:
        return win_rate_mc(hole_card, community_card, nb_simulation =_FAST_MC_SIMS)
    if street is not None and pot is not None and stack is not None:
        k = abstraction.dynamic_bucket_count(street, pot, stack)
        n = max(100, min(500, k * 25))  # k in [3,20] -> n in [100,500]
    elif time_budget > 0.30:
        n = 500
    elif time_budget > 0.15:
        n = 250
    else:
        n = 100
    return win_rate_mc(hole_card, community_card, nb_simulation = n)

# f2 = pots_ods_pressure odds of winning pot
# if win_rate > odds + pressure, < negative
def signed_pot_odds_pressure(pot_odds, win_rate):
    if pot_odds <= 0.0:
        return max(-1.0, min(1.0, 2.0 * (win_rate - 0.5)))
    if win_rate >= pot_odds:
        denom = max(1e-6, 1.0 - pot_odds)
        return max(0.0, min(1.0, (win_rate - pot_odds)/denom))
    denom = max(1e-6, pot_odds)
    return max(-1.0, min(0.0, (win_rate-pot_odds) / denom))

# evaluate move from features phi(state) = sum_j weights_j * phi_j(state)
def phi(features, win_rate, fold_equity, weights):
    f1 = win_rate
    f2 = signed_pot_odds_pressure(features["pot_odds"], win_rate)
    f3 = features["position"]
    f4 = fold_equity
    f5 = features["pot_commitment"]
    return (weights["w_hs"]* f1+ weights["w_pot_odds"]  * f2 + weights["w_position"]* f3 + weights["w_fold_equity"] *f4 + weights["w_pot_commit"] *f5)

# eval action from state value
def action_scores(features, win_rate, fold_equity, weights, valid_actions):
    base = phi(features, win_rate, fold_equity, weights)
    legal = {a["action"] for a in valid_actions}
    scores = {}
    if "fold" in legal:
        # lose chips on fold
        fold_penalty = max(0.0, base)
        scores["fold"] = weights["bias_fold"] - fold_penalty
    if "call" in legal:
        # same chips on call
        check_bonus = 0.15 if features["to_call"]== 0 else 0.0
        scores["call"] = weights["bias_call"] + base + check_bonus
    if "raise" in legal:
        # more chips on raise
        raise_extra = (weights["w_fold_equity"] * fold_equity *0.5+ weights["w_hs"] * max(0.0, win_rate - 0.5))
        scores["raise"] = weights["bias_raise"]+ base +raise_extra
    return scores