import math
import random

from pypokerengine.engine.card import Card

from . import cfr_abstraction as absn


STREETS = ("preflop", "flop", "turn", "river")
STREET_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
DEFAULT_PARAMS = {
    "max_combos": 40,
    "samples": 48,
    "policy_margin_uncertain": 0.10,
    "river_pot_units": 34.0,
    "river_to_call_units": 5.0,
    "turn_pot_units": 50.0,
    "turn_to_call_units": 8.0,
    "range_cache_limit": 128,
    "action_likelihood_floor": 0.025,
    "range_log_floor": -30.0,
    "credibility_samples": 24,
    "credibility_multiplier_base": 0.75,
    "credibility_multiplier_scale": 0.50,
    "max_fold_to_raise": 0.95,
    "turn_blend_base": 0.12,
    "river_blend_base": 0.18,
    "uncertain_blend_bonus": 0.05,
    "large_pot_blend_units": 36.0,
    "large_pot_blend_bonus": 0.04,
    "large_call_blend_units": 6.0,
    "large_call_blend_bonus": 0.04,
    "max_blend": 0.26,
}


class PublicBeliefSearch:
    def __init__(self, policy, rng=None, **params):
        self.policy = policy
        self.rng = rng or random.Random()
        self.params = _normalize_params(params)
        self.max_combos = int(max(8, self.params["max_combos"]))
        self.samples = int(max(16, self.params["samples"]))
        self._range_cache = {}

    def maybe_adjust_distribution(self, probs, valid_actions, hole_card, round_state, our_uuid):
        try:
            if not self._should_search(probs, valid_actions, round_state, our_uuid):
                return probs
            legal = [a for a in absn.ACTIONS if a in probs]
            if len(legal) <= 1:
                return probs

            our_ids = _cards_to_ids(hole_card)
            board_ids = _cards_to_ids(round_state.get("community_card", []) or [])
            known = set(our_ids) | set(board_ids)
            opp_uuid = _primary_opponent_uuid(round_state, our_uuid)
            if opp_uuid is None:
                return probs

            opp_range = self._range_for_actor(round_state, opp_uuid, known)
            if not opp_range:
                return probs
            our_credibility = self._public_line_credibility(
                round_state, our_uuid, tuple(our_ids), set(board_ids)
            )
            evs = self._action_evs(
                legal, valid_actions, hole_card, our_ids, board_ids,
                opp_range, round_state, our_uuid, opp_uuid, our_credibility,
            )
            search_probs = _softmax(evs)
            blend = self._blend_weight(probs, round_state, our_uuid)
            mixed = {}
            for action in legal:
                mixed[action] = (
                    (1.0 - blend) * max(0.0, float(probs.get(action, 0.0)))
                    + blend * max(0.0, float(search_probs.get(action, 0.0)))
                )
            return _normalize(mixed)
        except Exception:
            return probs

    def _should_search(self, probs, valid_actions, round_state, our_uuid):
        street = round_state.get("street", "preflop")
        if street == "preflop":
            return False
        small_blind = max(1.0, float(round_state.get("small_blind_amount", 10) or 10))
        pot_units = absn.pot_size(round_state) / small_blind
        to_call = _to_call(round_state, our_uuid)
        to_call_units = to_call / small_blind
        uncertain = _policy_margin(probs) <= self.params["policy_margin_uncertain"]
        allin_possible = (
            _raise_max(valid_actions) > 0
            and _raise_max(valid_actions) >= _our_stack(round_state, our_uuid)
        )
        return (
            (street == "river" and (
                pot_units >= self.params["river_pot_units"]
                or to_call_units >= self.params["river_to_call_units"]
                or allin_possible
                or uncertain
            ))
            or (street == "turn" and (
                pot_units >= self.params["turn_pot_units"]
                or to_call_units >= self.params["turn_to_call_units"]
                or (allin_possible and uncertain)
            ))
        )

    def _range_for_actor(self, round_state, actor_uuid, known_ids):
        cache_key = _range_cache_key(round_state, actor_uuid, known_ids)
        cached = self._range_cache.get(cache_key)
        if cached is not None:
            return cached
        combos = []
        deck = [cid for cid in range(1, 53) if cid not in known_ids]
        for i, c1 in enumerate(deck):
            for c2 in deck[i + 1:]:
                hole = (c1, c2)
                logp = self._line_loglikelihood(round_state, actor_uuid, hole)
                combos.append((logp, hole))
        if not combos:
            return []
        combos.sort(reverse=True, key=lambda item: item[0])
        keep = combos[: self.max_combos]
        weights = [math.exp(max(self.params["range_log_floor"], logp - keep[0][0])) for logp, _ in keep]
        total = sum(weights)
        if total <= 1e-12:
            result = [(hole, 1.0 / len(keep)) for _, hole in keep]
        else:
            result = [(hole, w / total) for w, (_, hole) in zip(weights, keep)]
        if len(self._range_cache) > self.params["range_cache_limit"]:
            self._range_cache.clear()
        self._range_cache[cache_key] = result
        return result

    def _line_loglikelihood(self, round_state, actor_uuid, hole_ids):
        histories = round_state.get("action_histories", {}) or {}
        board = round_state.get("community_card", []) or []
        logp = 0.0
        for street in STREETS:
            entries = histories.get(street, []) or []
            for idx, entry in enumerate(entries):
                if entry.get("uuid") != actor_uuid:
                    continue
                action = _norm_action(entry.get("action"))
                if action not in absn.ACTIONS:
                    continue
                pseudo = _prefix_state(round_state, street, idx, actor_uuid, board)
                facing = _is_facing(pseudo, actor_uuid)
                valid = _synthetic_valid_actions(facing)
                hole = _ids_to_cards(hole_ids)
                dist = self.policy.action_distribution(valid, hole, pseudo, actor_uuid)
                logp += math.log(max(self.params["action_likelihood_floor"], float(dist.get(action, 0.0))))
        return logp

    def _public_line_credibility(self, round_state, our_uuid, our_hole, board_ids):
        actual = self._line_loglikelihood(round_state, our_uuid, our_hole)
        deck = [cid for cid in range(1, 53) if cid not in board_ids]
        trials = []
        limit = min(int(self.params["credibility_samples"]), max(8, len(deck)))
        for _ in range(limit):
            if len(deck) < 2:
                break
            c1, c2 = self.rng.sample(deck, 2)
            if c1 == c2:
                continue
            trials.append(self._line_loglikelihood(round_state, our_uuid, (c1, c2)))
        if not trials:
            return 0.5
        below = sum(1 for v in trials if v <= actual)
        return min(1.0, max(0.0, below / float(len(trials))))

    def _action_evs(self, legal, valid_actions, hole_card, our_ids, board_ids,
                    opp_range, round_state, our_uuid, opp_uuid, credibility):
        pot = float(absn.pot_size(round_state))
        to_call = float(_to_call(round_state, our_uuid))
        raise_amount = float(_raise_min(valid_actions) or max(to_call, 0.0))
        evs = {}
        win_rate = self._sample_win_rate(our_ids, board_ids, opp_range)
        showdown_call = win_rate * (pot + to_call) - (1.0 - win_rate) * to_call

        for action in legal:
            if action == "fold":
                evs[action] = -to_call
            elif action == "call":
                evs[action] = showdown_call
            elif action == "raise":
                fold_prob = self._predicted_fold_to_raise(
                    round_state, opp_uuid, opp_range, credibility
                )
                called = win_rate * (pot + raise_amount) - (1.0 - win_rate) * raise_amount
                evs[action] = fold_prob * pot + (1.0 - fold_prob) * called
        return evs

    def _sample_win_rate(self, our_ids, board_ids, opp_range):
        wins = 0.0
        count = 0.0
        known_base = set(our_ids) | set(board_ids)
        for _ in range(self.samples):
            opp_ids = _weighted_choice(opp_range, self.rng)
            if opp_ids is None:
                continue
            known = known_base | set(opp_ids)
            deck = [cid for cid in range(1, 53) if cid not in known]
            need = max(0, 5 - len(board_ids))
            if len(deck) < need:
                continue
            runout = list(board_ids) + self.rng.sample(deck, need)
            result = absn.showdown_utility(list(our_ids), list(opp_ids), runout)
            wins += 1.0 if result > 0 else (0.5 if result == 0 else 0.0)
            count += 1.0
        return wins / count if count > 0.0 else 0.5

    def _predicted_fold_to_raise(self, round_state, opp_uuid, opp_range, credibility):
        pseudo = _state_after_our_raise(round_state, opp_uuid)
        valid = _synthetic_valid_actions(True)
        fold = 0.0
        for opp_ids, weight in opp_range:
            dist = self.policy.action_distribution(
                valid, _ids_to_cards(opp_ids), pseudo, opp_uuid
            )
            fold += weight * max(0.0, float(dist.get("fold", 0.0)))
        credibility_multiplier = (
            self.params["credibility_multiplier_base"]
            + self.params["credibility_multiplier_scale"]
            * max(0.0, min(1.0, credibility))
        )
        return max(0.0, min(self.params["max_fold_to_raise"], fold * credibility_multiplier))

    def _blend_weight(self, probs, round_state, our_uuid):
        street = round_state.get("street", "preflop")
        base = self.params["turn_blend_base"] if street == "turn" else self.params["river_blend_base"]
        if _policy_margin(probs) <= self.params["policy_margin_uncertain"]:
            base += self.params["uncertain_blend_bonus"]
        small_blind = max(1.0, float(round_state.get("small_blind_amount", 10) or 10))
        if absn.pot_size(round_state) / small_blind >= self.params["large_pot_blend_units"]:
            base += self.params["large_pot_blend_bonus"]
        if _to_call(round_state, our_uuid) / small_blind >= self.params["large_call_blend_units"]:
            base += self.params["large_call_blend_bonus"]
        return max(0.0, min(self.params["max_blend"], base))


def _prefix_state(round_state, street, stop_idx, next_uuid, current_board):
    histories = {}
    for s in STREETS:
        entries = (round_state.get("action_histories", {}) or {}).get(s, []) or []
        if STREETS.index(s) < STREETS.index(street):
            histories[s] = list(entries)
        elif s == street:
            histories[s] = list(entries[:stop_idx])
        else:
            histories[s] = []
    pseudo = dict(round_state)
    pseudo["street"] = street
    pseudo["action_histories"] = histories
    pseudo["community_card"] = list(current_board[: STREET_BOARD.get(street, 5)])
    pseudo["next_player"] = _seat_index(pseudo, next_uuid)
    return pseudo


def _state_after_our_raise(round_state, next_uuid):
    pseudo = dict(round_state)
    histories = {
        s: list((round_state.get("action_histories", {}) or {}).get(s, []) or [])
        for s in STREETS
    }
    street = round_state.get("street", "preflop")
    seats = round_state.get("seats", []) or []
    next_idx = _seat_index(round_state, next_uuid)
    our_uuid = None
    for seat in seats:
        uuid = seat.get("uuid")
        if uuid and uuid != next_uuid:
            our_uuid = uuid
            break
    amount = _max_street_bet(round_state, street) + max(1, int(round_state.get("small_blind_amount", 10) or 10) * 2)
    histories.setdefault(street, []).append({
        "action": "RAISE",
        "amount": amount,
        "add_amount": max(1, int(round_state.get("small_blind_amount", 10) or 10) * 2),
        "paid": max(1, int(round_state.get("small_blind_amount", 10) or 10) * 2),
        "uuid": our_uuid,
    })
    pseudo["action_histories"] = histories
    pseudo["next_player"] = next_idx
    return pseudo


def _primary_opponent_uuid(round_state, our_uuid):
    for seat in round_state.get("seats", []) or []:
        uuid = seat.get("uuid")
        if uuid and uuid != our_uuid and seat.get("state") in ("participating", "allin"):
            return uuid
    for seat in round_state.get("seats", []) or []:
        uuid = seat.get("uuid")
        if uuid and uuid != our_uuid:
            return uuid
    return None


def _range_cache_key(round_state, actor_uuid, known_ids):
    histories = round_state.get("action_histories", {}) or {}
    hist_sig = []
    for street in STREETS:
        for entry in histories.get(street, []) or []:
            hist_sig.append((
                street,
                entry.get("uuid"),
                entry.get("action"),
                int(entry.get("amount", 0) or 0),
            ))
    return (
        int(round_state.get("round_count", 0) or 0),
        round_state.get("street", "preflop"),
        actor_uuid,
        tuple(sorted(known_ids)),
        tuple(hist_sig),
    )


def _seat_index(round_state, uuid):
    for i, seat in enumerate(round_state.get("seats", []) or []):
        if seat.get("uuid") == uuid:
            return i
    return 0


def _is_facing(round_state, uuid):
    return _to_call(round_state, uuid) > 0


def _to_call(round_state, uuid):
    street = round_state.get("street", "preflop")
    seats = round_state.get("seats", []) or []
    idx = _seat_index(round_state, uuid)
    bets = [0 for _ in seats] or [0, 0]
    uuid_to_pos = {seat.get("uuid"): i for i, seat in enumerate(seats)}
    for entry in (round_state.get("action_histories", {}) or {}).get(street, []) or []:
        action = (entry.get("action") or "").upper()
        pos = uuid_to_pos.get(entry.get("uuid"))
        if pos is None or action == "FOLD":
            continue
        if action in ("CALL", "RAISE", "SMALLBLIND", "BIGBLIND"):
            bets[pos] = max(bets[pos], int(entry.get("amount", 0) or 0))
    if idx >= len(bets):
        return 0
    return max(0, max(bets or [0]) - bets[idx])


def _max_street_bet(round_state, street):
    max_bet = 0
    for entry in (round_state.get("action_histories", {}) or {}).get(street, []) or []:
        if (entry.get("action") or "").upper() != "FOLD":
            max_bet = max(max_bet, int(entry.get("amount", 0) or 0))
    return max_bet


def _synthetic_valid_actions(facing):
    call_amount = 10 if facing else 0
    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": call_amount},
        {"action": "raise", "amount": {"min": 20, "max": 1000}},
    ]


def _raise_min(valid_actions):
    for action in valid_actions:
        if action.get("action") != "raise":
            continue
        amount = action.get("amount")
        if isinstance(amount, dict):
            return amount.get("min", 0) or 0
        return amount or 0
    return 0


def _raise_max(valid_actions):
    for action in valid_actions:
        if action.get("action") != "raise":
            continue
        amount = action.get("amount")
        if isinstance(amount, dict):
            return amount.get("max", 0) or 0
        return amount or 0
    return 0


def _our_stack(round_state, our_uuid):
    for seat in round_state.get("seats", []) or []:
        if seat.get("uuid") == our_uuid:
            return int(seat.get("stack", 0) or 0)
    return 0


def _policy_margin(probs):
    vals = sorted((max(0.0, float(v)) for v in probs.values()), reverse=True)
    if len(vals) < 2:
        return 1.0
    return vals[0] - vals[1]


def _cards_to_ids(cards):
    return [Card.from_str(c).to_id() for c in cards]


def _ids_to_cards(ids):
    return [str(absn.CARD_BY_ID[cid]) for cid in ids]


def _norm_action(action):
    action = (action or "").lower()
    if action in ("smallblind", "bigblind", "ante"):
        return None
    return action if action in absn.ACTIONS else None


def _weighted_choice(weighted, rng):
    total = sum(max(0.0, float(w)) for _, w in weighted)
    if total <= 1e-12:
        return weighted[0][0] if weighted else None
    pick = rng.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += max(0.0, float(weight))
        if pick <= acc:
            return value
    return weighted[-1][0]


def _softmax(evs):
    if not evs:
        return {}
    scale = max(1.0, max(abs(v) for v in evs.values()) / 4.0)
    best = max(evs.values())
    raw = {a: math.exp((v - best) / scale) for a, v in evs.items()}
    return _normalize(raw)


def _normalize_params(params):
    merged = dict(DEFAULT_PARAMS)
    for key, value in (params or {}).items():
        if key not in merged:
            continue
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            pass
    for key in ("max_combos", "samples", "range_cache_limit", "credibility_samples"):
        merged[key] = int(max(1, round(merged[key])))
    merged["action_likelihood_floor"] = max(1e-6, min(0.34, merged["action_likelihood_floor"]))
    merged["max_blend"] = max(0.0, min(1.0, merged["max_blend"]))
    merged["max_fold_to_raise"] = max(0.0, min(1.0, merged["max_fold_to_raise"]))
    return merged


def _normalize(probs):
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 1e-12:
        n = max(1, len(probs))
        return {a: 1.0 / n for a in probs}
    return {a: max(0.0, float(v)) / total for a, v in probs.items()}
