from pypokerengine.utils.card_utils import (estimate_hole_card_win_rate, gen_cards)

RANK_ORDER = "23456789TJQKA"
def _card_rank(card_str):
    return card_str[1].upper()

def _card_suit(card_str):
    return card_str[0].upper()

# classify game preflop
def preflop(hole_card):
    if len(hole_card) != 2:
        return "??"
    r1, r2 = _card_rank(hole_card[0]), _card_rank(hole_card[1])
    s1, s2 = _card_suit(hole_card[0]), _card_suit(hole_card[1])
    if r1 == r2:
        return r1 + r2  #ex. AA, 77
    # rankcards inorder
    if RANK_ORDER.index(r1) < RANK_ORDER.index(r2):
        r1, r2 = r2, r1
    return r1 + r2 + ("s" if s1 ==s2 else "o")

# small pot use k=5 buckets, big use 15-20
def dynamic_bucket_count(street, pot, stack):
    street_idx = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(street, 1)
    # +2 per next street, +5 if big pot
    k = 5 + 2 * (street_idx - 1)
    big_pot = pot > 0 and stack> 0 and (float(pot)/float(stack)) > 0.3
    if big_pot:
        k += 5
    return max(3, min(k, 20))

# (hole, community) -> bucket [0, k)
# by MC estimate of hand strength https://www.cs.cmu.edu/~sandholm/hierarchical.aamas15.pdf
def postflop_bucket(hole_card, community, k, nb_simulation =120):
    if k <= 1:
        return 0
    hand_str = estimate_hole_card_win_rate(nb_simulation=nb_simulation, nb_player =2,hole_card= gen_cards(hole_card), community_card = gen_cards(community) if community else [])
    bucket = int(hand_str * k)
    if bucket >= k:
        bucket = k - 1
    return bucket