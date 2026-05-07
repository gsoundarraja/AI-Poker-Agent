import json
import os
import random
from bisect import bisect_right

from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator


# make buckets of states
ACTIONS = ("fold", "call", "raise")
INFOSET_STREET = 0
INFOSET_PLAYER = 1
INFOSET_POSITION = 2
INFOSET_CARD_BUCKET = 3
INFOSET_POT_BUCKET = 4
INFOSET_CALL_BUCKET = 5
INFOSET_STREET_RAISES = 6
INFOSET_TOTAL_RAISES = 7
INFOSET_MAX_BET = 8
INFOSET_FACING = 9
INFOSET_LENGTH = 10
INDEX_TO_STREET = ("preflop", "flop", "turn", "river")
RANK_ORDER = "23456789TJQKA"
CARD_BY_ID = [None] + [Card.from_id(cid) for cid in range(1, 53)]
RANK_BY_ID = [0] + [CARD_BY_ID[cid].rank for cid in range(1, 53)]
SUIT_BY_ID = [None] + [CARD_BY_ID[cid].suit for cid in range(1, 53)]


def normalize_probs(probs):
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 1e-12:
        n = max(1, len(probs))
        return {a: 1.0 / n for a in probs}
    return {a: max(0.0, float(v)) / total for a, v in probs.items()}


def legal_action_names(valid_actions):
    legal = [a["action"] for a in valid_actions]
    call_amount = None
    for action in valid_actions:
        if action.get("action") == "call":
            call_amount = action.get("amount")
            break
    # free check
    if isinstance(call_amount, (int, float)) and call_amount <= 0 and "call" in legal:
        legal = [a for a in legal if a != "fold"]
    return legal


def mask_probs(probs, valid_actions):
    legal = legal_action_names(valid_actions)
    return normalize_probs({a: max(0.0, float(probs.get(a, 0.0))) for a in legal})


def parse_infoset_key(key):
    try:
        vals = tuple(int(x) for x in key.split("|"))
    except Exception:
        return None
    return vals if len(vals) == INFOSET_LENGTH else None


def load_abstraction(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default_abstraction()


def save_abstraction(path, metadata):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def default_abstraction():
    classes = all_preflop_classes()
    ranked = sorted(classes, key=_preflop_class_score)
    bucket_map = {}
    for i, key in enumerate(ranked):
        bucket_map[key] = min(7, int(i * 8 / max(1, len(ranked))))
    return {
        "version": 1,
        "card_buckets": 8,
        "pot_buckets": 6,
        "to_call_buckets": 5,
        "preflop_bucket_map": bucket_map,
        "postflop_thresholds": {
            "flop": [4096, 8192, 12288, 16384, 20480, 24576, 28672],
            "turn": [4096, 8192, 12288, 16384, 20480, 24576, 28672],
            "river": [4096, 8192, 12288, 16384, 20480, 24576, 28672],
        },
        "pot_thresholds": [4.0, 8.0, 12.0, 20.0, 32.0],
        "to_call_thresholds": [0.5, 1.5, 2.5, 4.5],
        "source": "default_quantile_seed",
    }


# learn bucket abstraction from random
def learn_abstraction(samples, seed, card_buckets=8, pot_buckets=6, to_call_buckets=5):
    rng = random.Random(seed)
    preflop = {key: [0.0, 0.0] for key in all_preflop_classes()}
    post = {"flop": [], "turn": [], "river": []}
    pot_units = []
    to_call_units = []
    for _ in range(max(1, samples)):
        deck = list(range(1, 53))
        rng.shuffle(deck)
        h0 = deck[:2]
        h1 = deck[2:4]
        board = deck[4:9]

        key = preflop_class_from_ids(h0)
        result = showdown_utility(h0, h1, board)
        preflop[key][0] += 1.0 if result > 0 else (0.5 if result == 0 else 0.0)
        preflop[key][1] += 1.0

        for street, n_cards in (("flop", 3), ("turn", 4), ("river", 5)):
            post[street].append(hand_value(h0, board[:n_cards]))

        trace = sample_betting_trace_units(rng)
        pot_units.extend(trace["pot"])
        to_call_units.extend(trace["to_call"])

    class_values = {}
    for key, (wins, count) in preflop.items():
        if count:
            class_values[key] = wins / count
        else:
            class_values[key] = _preflop_class_score(key)
    sorted_classes = sorted(class_values, key=lambda k: class_values[k])
    preflop_bucket_map = {}
    for i, key in enumerate(sorted_classes):
        preflop_bucket_map[key] = min(card_buckets - 1, int(i * card_buckets / max(1, len(sorted_classes))))

    post_thresholds = {}
    for street, values in post.items():
        post_thresholds[street] = quantile_thresholds(values, card_buckets)

    return {
        "version": 1,
        "card_buckets": card_buckets,
        "pot_buckets": pot_buckets,
        "to_call_buckets": to_call_buckets,
        "preflop_bucket_map": preflop_bucket_map,
        "preflop_class_values": class_values,
        "postflop_thresholds": post_thresholds,
        "pot_thresholds": quantile_thresholds(pot_units, pot_buckets),
        "to_call_thresholds": quantile_thresholds(to_call_units, to_call_buckets),
        "source": "sampled_rollout_quantiles",
        "samples": samples,
        "seed": seed,
    }


def quantile_thresholds(values, bucket_count):
    if bucket_count <= 1:
        return []
    if not values:
        return []
    values = sorted(values)
    thresholds = []
    n = len(values)
    for i in range(1, bucket_count):
        idx = min(n - 1, max(0, int(i * n / bucket_count)))
        thresholds.append(values[idx])
    deduped = []
    for value in thresholds:
        if not deduped or value > deduped[-1]:
            deduped.append(value)
    return deduped


def bucket_value(value, thresholds, max_bucket=None):
    bucket = bisect_right(thresholds or [], value)
    if max_bucket is not None:
        return min(max_bucket, bucket)
    return bucket


def card_bucket(metadata, hole_ids, community_ids):
    street_idx = street_index_from_board_len(len(community_ids))
    max_bucket = int(metadata.get("card_buckets", 8)) - 1
    if street_idx == 0:
        # card -> hand class
        key = preflop_class_from_ids(hole_ids)
        bucket_map = metadata.get("preflop_bucket_map", {})
        if key in bucket_map:
            return int(bucket_map[key])
        return min(max_bucket, int(_preflop_class_score(key) * (max_bucket + 1)))
    street = INDEX_TO_STREET[street_idx]
    thresholds = metadata.get("postflop_thresholds", {}).get(street, [])
    return bucket_value(hand_value(hole_ids, community_ids), thresholds, max_bucket)


def infoset_key(metadata, player, hole_ids, community_ids, pot, to_call, street_bets,
                raises_this_street, total_raises, is_small_blind, small_blind):
    sb = float(max(1, small_blind))
    street_idx = street_index_from_board_len(len(community_ids))
    c_bucket = card_bucket(metadata, hole_ids, community_ids)
    # money in pot
    pot_bucket = bucket_value(
        float(pot) / sb,
        metadata.get("pot_thresholds", []),
        int(metadata.get("pot_buckets", 6)) - 1,
    )
    call_bucket = bucket_value(
        float(to_call) / sb,
        metadata.get("to_call_thresholds", []),
        int(metadata.get("to_call_buckets", 5)) - 1,
    )
    facing = 1 if to_call > 0 else 0
    max_bet_units = int(round(max(street_bets) / sb)) if street_bets else 0
    pos = 1 if is_small_blind else 0
    return "|".join(str(x) for x in (
        street_idx,
        player,
        pos,
        c_bucket,
        pot_bucket,
        call_bucket,
        min(4, raises_this_street),
        min(8, total_raises),
        min(20, max_bet_units),
        facing,
    ))

# game state -> key for CFR
def runtime_infoset(metadata, round_state, hole_card, player_uuid):
    seats = round_state.get("seats", []) or []
    player_idx = 0
    for i, seat in enumerate(seats):
        if seat.get("uuid") == player_uuid:
            player_idx = i
            break
    street = round_state.get("street", "preflop")
    street_bets = [0 for _ in seats] or [0, 0]
    raises_this = 0
    total_raises = 0
    uuid_to_pos = {seat.get("uuid"): i for i, seat in enumerate(seats)}
    # get how much each player bet ts street
    for s_name, entries in (round_state.get("action_histories", {}) or {}).items():
        for entry in entries or []:
            action = (entry.get("action") or "").upper()
            if action == "RAISE":
                total_raises += 1
                if s_name == street:
                    raises_this += 1
            if s_name == street and action != "FOLD":
                pos = uuid_to_pos.get(entry.get("uuid"))
                if pos is not None:
                    street_bets[pos] = max(street_bets[pos], entry.get("amount", 0) or 0)
    if player_idx >= len(street_bets):
        player_idx = 0
    to_call = max(0, max(street_bets) - street_bets[player_idx])
    small_blind = round_state.get("small_blind_amount", 10) or 10
    return infoset_key(
        metadata=metadata,
        player=0,
        hole_ids=[Card.from_str(c).to_id() for c in hole_card],
        community_ids=[Card.from_str(c).to_id() for c in round_state.get("community_card", [])],
        pot=pot_size(round_state),
        to_call=to_call,
        street_bets=street_bets,
        raises_this_street=raises_this,
        total_raises=total_raises,
        is_small_blind=(player_idx == round_state.get("small_blind_pos", -1)),
        small_blind=small_blind,
    )


def pot_size(round_state):
    pot = round_state.get("pot", {}) or {}
    total = (pot.get("main", {}) or {}).get("amount", 0) or 0
    for side in pot.get("side", []) or []:
        total += side.get("amount", 0) or 0
    return total


def street_index_from_board_len(n):
    if n <= 0:
        return 0
    if n == 3:
        return 1
    if n == 4:
        return 2
    return 3


def hand_value(hole_ids, community_ids):
    return HandEvaluator.eval_hand(cards_from_ids(hole_ids), cards_from_ids(community_ids))


def showdown_utility(hole0, hole1, board):
    v0 = hand_value(hole0, board)
    v1 = hand_value(hole1, board)
    if v0 > v1:
        return 1
    if v1 > v0:
        return -1
    return 0


def cards_from_ids(ids):
    return [CARD_BY_ID[cid] for cid in ids]


# name preflop hand
def preflop_class_from_ids(card_ids):
    r1, r2 = _rank_char(RANK_BY_ID[card_ids[0]]), _rank_char(RANK_BY_ID[card_ids[1]])
    suited = SUIT_BY_ID[card_ids[0]] == SUIT_BY_ID[card_ids[1]]
    if r1 == r2:
        return r1 + r2
    if RANK_ORDER.index(r1) < RANK_ORDER.index(r2):
        r1, r2 = r2, r1
    return r1 + r2 + ("s" if suited else "o")


def all_preflop_classes():
    classes = []
    for i, r1 in enumerate(RANK_ORDER):
        for j, r2 in enumerate(RANK_ORDER):
            if i < j:
                continue
            hi = r1 if i >= j else r2
            lo = r2 if i >= j else r1
            if hi == lo:
                classes.append(hi + lo)
            else:
                classes.append(hi + lo + "s")
                classes.append(hi + lo + "o")
    return sorted(set(classes))


def _rank_char(rank):
    return "23456789TJQKA"[rank - 2]


# score preflop class
def _preflop_class_score(key):
    ranks = key[:2]
    r1 = RANK_ORDER.index(ranks[0]) + 2
    r2 = RANK_ORDER.index(ranks[1]) + 2
    pair = ranks[0] == ranks[1]
    suited = key.endswith("s")
    gap = abs(r1 - r2)
    score = (max(r1, r2) + min(r1, r2) * 0.55) / 22.0
    if pair:
        score += 0.30 + r1 / 40.0
    if suited:
        score += 0.045
    if gap <= 2 and not pair:
        score += 0.03
    if gap >= 5:
        score -= 0.05
    return max(0.0, min(1.0, score))


# fake random betting pot sizes to learn buckets
def sample_betting_trace_units(rng):
    pot = 3.0
    traces = {"pot": [pot], "to_call": [1.0]}
    for street in range(4):
        bets = [1.0, 2.0] if street == 0 else [0.0, 0.0]
        raises = 0
        acted = [False, False]
        player = 0
        inc = 2.0 if street <= 1 else 4.0
        for _ in range(10):
            to_call = max(0.0, max(bets) - bets[player])
            traces["pot"].append(pot)
            traces["to_call"].append(to_call)
            r = rng.random()
            if to_call > 0 and r < 0.18:
                break
            if raises < 4 and r > 0.68:
                new_amount = max(bets) + inc
                pot += max(0.0, new_amount - bets[player])
                bets[player] = new_amount
                raises += 1
                acted = [False, False]
                acted[player] = True
            else:
                pot += to_call
                bets[player] += to_call
                acted[player] = True
            if acted[0] and acted[1] and abs(bets[0] - bets[1]) < 1e-9:
                break
            player = 1 - player
    return traces
