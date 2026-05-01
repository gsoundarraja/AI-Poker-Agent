# expectiminimax https://en.wikipedia.org/wiki/Expectiminimax
# agent is max, opp is MIN

import math
import random

def _ev_action(action, pot, to_call, inc, win_rate, opp_dist):
    # fold = 0, call = wr * pot - (1-wr)*call
    # raise = exp over opp distribution (fold, call, raise)
    if action =="fold":
        return 0.0

    if action == "call":
        return win_rate * pot -  (1.0 - win_rate) *to_call
    if action == "raise":
        # pay
        R = to_call + inc

        p_fold  = opp_dist.get("fold", 0.35)
        p_call  = opp_dist.get("call", 0.45)
        p_raise = opp_dist.get("raise", 0.20)
        ev_opp_fold = float(pot)
        # call * R
        ev_opp_call = win_rate * (pot + inc) - (1.0 - win_rate) * R
        # 3bet first raise before flop
        ev_if_fold_to_3bet = -R
        # call * (R+inc) or 2*inc
        ev_if_call_3bet = (win_rate * (pot + 2 * inc)- (1.0 - win_rate) * (R + inc))
        ev_opp_raise = max(ev_if_fold_to_3bet, ev_if_call_3bet)
        return p_fold * ev_opp_fold + p_call * ev_opp_call + p_raise * ev_opp_raise
    return 0.0

# best = highest expectiminimax
def choose_best_action(features, win_rate, opp_dist, valid_actions, budget= 0.4):
    legal = {a["action"] for a in valid_actions}
    pot = features["pot"]
    to_call = features["to_call"]
    inc = features["raise_increment"]

    scores = {}
    for action in ("fold", "call", "raise"):
        if action in legal:
            scores[action] = _ev_action(action, pot, to_call,inc, win_rate, opp_dist)
    if not scores:
        return "fold"
    # nvr fold free
    if to_call == 0 and "fold" in scores:
        del scores["fold"]

    best_score = max(scores.values())

    # softmax stochastic similar actions within 15 chips
    candidates = {a: v  for a, v in scores.items() if (best_score - v)<= 15.0}
    if len(candidates) == 1:
        return next(iter(candidates))
    weights = {a: math.exp((v -best_score) / 6.0) for a, v in candidates.items()}
    total = sum(weights.values())
    r = random.random()* total
    cum = 0.0
    for a, w in weights.items():
        cum += w
        if r <= cum:
            return a
    return max(candidates, key=lambda k: candidates[k])
