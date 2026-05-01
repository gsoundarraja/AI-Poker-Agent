# expectiminimax https://en.wikipedia.org/wiki/Expectiminimax
# agent is max, opp is MIN

from . import evaluation as ev

# chip weight for trained eval phi(s) + bonus from exp val
PHI_SCALE = 50.0

def _phi_bonus(action, features, win_rate, fold_eq, weights):
    base = ev.phi(features, win_rate, fold_eq, weights)
    bias = weights.get("bias_" + action, 0.0)
    if action == "fold":
        # - fold when state is good
        return PHI_SCALE * (bias - max(0.0, base))
    if action == "raise":
        # raise reward good hand/equity
        extra = weights["w_fold_equity"] * fold_eq * 0.5 + weights["w_hs"] * max(0.0, win_rate - 0.5)
        return PHI_SCALE * (bias + base + extra)
    # call
    check_bonus = 0.15 if features.get("to_call", 0) == 0 else 0.0
    return PHI_SCALE * (bias + base + check_bonus)

def _ev_action(action, features, win_rate, opp_dist, weights):
    # fold = 0 call = wr * pot - (1-wr)*call
    # raise = exp over opp distribution (fold, call, raise)
    pot     = features["pot"]
    to_call = features["to_call"]
    inc     = features["raise_increment"]
    fold_eq = opp_dist.get("fold", 0.35)

    if action =="fold":
        base = 0.0

    elif action == "call":
        base = win_rate * pot -  (1.0 - win_rate) *to_call
    elif action == "raise":
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
        base = p_fold * ev_opp_fold + p_call * ev_opp_call + p_raise * ev_opp_raise
    else:
        return 0.0

    return base + _phi_bonus(action, features, win_rate, fold_eq, weights)

# best = highest expectiminimax
def choose_best_action(features, win_rate, opp_dist, valid_actions, weights):
    legal = {a["action"] for a in valid_actions}
    to_call = features["to_call"]

    scores = {}
    for action in ("fold", "call", "raise"):
        if action in legal:
            scores[action] = _ev_action(action, features, win_rate, opp_dist, weights)
    if not scores:
        return "fold"
    # tie break prefer call
    best_action = max(scores, key=lambda a: (scores[a], 1 if a == "call" else 0))

    # raise if clearly strong
    if ("raise" in scores and "call" in scores and scores["raise"] >= scores["call"] - 1.0 and win_rate > 0.6):
        best_action = "raise"

    if best_action == "fold" and to_call == 0 and "call" in scores:
        best_action = "call"

    return best_action