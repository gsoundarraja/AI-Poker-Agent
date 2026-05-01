import argparse
import os
import pickle
import sys
import time
# precompute map preflop "AA" -> win-rate on uniform-random
# find pypokerengine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

def _hand_str(cls):
    if len(cls) == 2:
        r = cls[0]
        # suit + rank, eg QH = queen of hearts
        # T=10, J(jack) =11, Q(queen) = 12, K(king)=13, A(ace)=14
        return ["H" + r, "D" + r]
    r1, r2, kind = cls[0], cls[1], cls[2]
    if kind == "s":
        # same suit
        return ["H" + r1, "H" + r2]
    return ["H" + r1, "D" + r2]

# enumerate tottal possible 2 card hands into classes of suited or offsuit pairs
def enum_classes():
    classes = []
    ranks = list("23456789TJQKA")
    # pairs
    for r in ranks:
        classes.append(r + r)
    #non-pairs
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            lo, hi = ranks[i], ranks[j]
            classes.append(hi + lo + "s")
            classes.append(hi + lo + "o")
    return classes

def main(nb_simulation=5000):
    t0 = time.time()
    classes = enum_classes()
    table = {}
    print(f'running precomp on {len(classes)} classes {nb_simulation} simuls each')
    for i, cls in enumerate(classes):
        hand_strs = _hand_str(cls)
        wr = estimate_hole_card_win_rate(nb_simulation=nb_simulation,nb_player= 2, hole_card = gen_cards(hand_strs),community_card =[])
        table[cls] = wr
        if (i + 1) % 20 == 0:
            print(f"{i+1} out of {len(classes)} in {(time.time()-t0):.2f}s")

    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data")
    os.makedirs(out_dir, exist_ok =True)
    out_path = os.path.join(out_dir, "preflop_hs.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(table, f)
    print(f"wrote {len(table)} to {out_path} in {(time.time()-t0):.2f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type =int, default= 5000)
    args = ap.parse_args()
    main(nb_simulation = args.sims)