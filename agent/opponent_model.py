from collections import defaultdict

from . import cfr_abstraction as absn


STREETS = ("preflop", "flop", "turn", "river")
DEFAULT_PARAMS = {
    "prior": 1.0,
    "min_samples": 4.0,
    "max_shift": 0.45,
    "opp_raise_fold_scale": 1.8,
    "opp_raise_call_scale": 1.4,
    "opp_raise_reraise_scale": 0.7,
    "opp_raise_deterrence": 0.15,
    "opp_raise_call_bonus": 0.15,
    "vs_our_raise_fold_scale": 1.0,
    "vs_our_raise_reraise_deterrence": 0.5,
    "image_raise_surprise_scale": 0.5,
    "image_fold_reduce_fold_scale": 0.7,
    "image_fold_call_scale": 0.4,
    "min_multiplier": 0.45,
    "max_multiplier": 1.70,
}


def normalize_params(params=None):
    merged = dict(DEFAULT_PARAMS)
    for key, value in (params or {}).items():
        if key in merged:
            merged[key] = float(value)
    if merged["min_multiplier"] <= 0.0:
        merged["min_multiplier"] = DEFAULT_PARAMS["min_multiplier"]
    if merged["max_multiplier"] < merged["min_multiplier"]:
        merged["max_multiplier"] = merged["min_multiplier"]
    return merged


class OnlineOpponentModel:
    def __init__(self, **params):
        self.params = normalize_params(params)
        self.prior = self.params["prior"]
        self.min_samples = self.params["min_samples"]
        self.max_shift = self.params["max_shift"]
        self.counts = defaultdict(self._new_counts)
        self._seen = set()

    def reset(self):
        self.counts.clear()
        self._seen.clear()

    def observe_action(self, new_action, round_state, our_uuid):
        action = _norm_action(new_action.get("action"))
        actor = new_action.get("player_uuid") or new_action.get("uuid")
        if action not in absn.ACTIONS or not actor:
            return

        street, index = self._locate_action(new_action, round_state)
        amount = int(new_action.get("amount", 0) or 0)
        event_id = (
            int(round_state.get("round_count", 0) or 0),
            street,
            index,
            actor,
            action,
            amount,
        )
        if event_id in self._seen:
            return
        self._seen.add(event_id)

        facing, facing_our_raise = self._context_before_action(
            round_state, street, index, actor, our_uuid
        )
        role = "self" if actor == our_uuid else "opp"
        self._increment(role, street, facing, facing_our_raise, action)

    def adjust_distribution(self, probs, round_state, our_uuid):
        if not probs or our_uuid is None:
            return probs

        street = round_state.get("street", "preflop")
        if street not in STREETS:
            street = "river"
        facing = self._current_facing(round_state, our_uuid)
        adjusted = {a: max(0.0, float(p)) for a, p in probs.items()}
        multipliers = {a: 1.0 for a in adjusted}

        vs_our_raise, n_vs_raise = self.distribution(
            "opp", street, facing=True, facing_our_raise=True
        )
        conf_vs_raise = self._confidence(n_vs_raise)
        if "raise" in multipliers:
            fold_excess = vs_our_raise["fold"] - (1.0 / 3.0)
            raise_excess = vs_our_raise["raise"] - (1.0 / 3.0)
            multipliers["raise"] *= (
                1.0
                + self.params["vs_our_raise_fold_scale"] * self.max_shift
                * conf_vs_raise * fold_excess * 2.0
            )
            multipliers["raise"] *= (
                1.0
                - self.params["vs_our_raise_reraise_deterrence"] * self.max_shift
                * conf_vs_raise * max(0.0, raise_excess) * 2.0
            )

        opp_dist, n_opp = self.distribution("opp", street)
        opp_all, n_opp_all = self.distribution("opp", None)
        local_weight = self._confidence(n_opp)
        opp_raise_rate = (
            local_weight * opp_dist["raise"]
            + (1.0 - local_weight) * opp_all["raise"]
        )
        conf_opp = self._confidence(n_opp + 0.5 * n_opp_all)
        raise_excess = max(0.0, opp_raise_rate - (1.0 / 3.0))

        if facing and raise_excess > 0.0:
            if "fold" in multipliers:
                multipliers["fold"] *= (
                    1.0
                    - self.params["opp_raise_fold_scale"] * self.max_shift
                    * conf_opp * raise_excess
                )
            if "call" in multipliers:
                multipliers["call"] *= (
                    1.0
                    + self.params["opp_raise_call_scale"] * self.max_shift
                    * conf_opp * raise_excess
                )
            if "raise" in multipliers:
                multipliers["raise"] *= (
                    1.0
                    + self.params["opp_raise_reraise_scale"] * self.max_shift
                    * conf_opp * raise_excess
                )

        if "raise" in multipliers:
            multipliers["raise"] *= (
                1.0
                - self.params["opp_raise_deterrence"] * self.max_shift
                * conf_opp * raise_excess * 2.0
            )
        if "call" in multipliers:
            multipliers["call"] *= (
                1.0
                + self.params["opp_raise_call_bonus"] * self.max_shift
                * conf_opp * raise_excess * 2.0
            )

        image_dist, n_image = self.distribution("self", street, facing=facing)
        image_facing, n_image_facing = self.distribution("self", None, facing=True)
        conf_image = self._confidence(n_image)
        if "raise" in multipliers:
            raise_surprise = (1.0 / 3.0) - image_dist["raise"]
            multipliers["raise"] *= (
                1.0
                + self.params["image_raise_surprise_scale"] * self.max_shift
                * conf_image * raise_surprise * 2.0
            )
        if facing and n_image_facing > 0:
            conf_fold_image = self._confidence(n_image_facing)
            fold_image = max(0.0, image_facing["fold"] - (1.0 / 3.0))
            if "fold" in multipliers:
                multipliers["fold"] *= (
                    1.0
                    - self.params["image_fold_reduce_fold_scale"] * self.max_shift
                    * conf_fold_image * fold_image * 2.0
                )
            if "call" in multipliers:
                multipliers["call"] *= (
                    1.0
                    + self.params["image_fold_call_scale"] * self.max_shift
                    * conf_fold_image * fold_image * 2.0
                )

        for action in adjusted:
            adjusted[action] *= _clamp(
                multipliers[action],
                self.params["min_multiplier"],
                self.params["max_multiplier"],
            )
        return _normalize(adjusted)

    def distribution(self, role, street=None, facing=None, facing_our_raise=None):
        totals = self._new_counts()
        observed = 0.0
        for key, counts in self.counts.items():
            k_role, k_street, k_facing, k_facing_our_raise = key
            if k_role != role:
                continue
            if street is not None and k_street != street:
                continue
            if facing is not None and k_facing != int(bool(facing)):
                continue
            if facing_our_raise is not None and k_facing_our_raise != int(bool(facing_our_raise)):
                continue
            for action in absn.ACTIONS:
                totals[action] += counts[action] - self.prior
                observed += counts[action] - self.prior
        return _normalize(totals), max(0.0, observed)

    def _increment(self, role, street, facing, facing_our_raise, action):
        key = (role, street, int(bool(facing)), int(bool(facing_our_raise)))
        self.counts[key][action] += 1.0

    def _new_counts(self):
        return {action: self.prior for action in absn.ACTIONS}

    def _confidence(self, samples):
        return min(1.0, max(0.0, float(samples)) / max(1.0, self.min_samples))

    def _locate_action(self, new_action, round_state):
        target_uuid = new_action.get("player_uuid") or new_action.get("uuid")
        target_action = _norm_action(new_action.get("action"))
        target_amount = int(new_action.get("amount", 0) or 0)
        histories = round_state.get("action_histories", {}) or {}
        best = (round_state.get("street", "preflop"), -1)
        for street in STREETS:
            for idx, entry in enumerate(histories.get(street, []) or []):
                if entry.get("uuid") != target_uuid:
                    continue
                if _norm_action(entry.get("action")) != target_action:
                    continue
                if int(entry.get("amount", 0) or 0) != target_amount:
                    continue
                best = (street, idx)
        return best

    def _context_before_action(self, round_state, street, index, actor_uuid, our_uuid):
        histories = round_state.get("action_histories", {}) or {}
        entries = histories.get(street, []) or []
        if index < 0:
            index = len(entries)
        seats = round_state.get("seats", []) or []
        uuid_to_pos = {seat.get("uuid"): i for i, seat in enumerate(seats)}
        actor_pos = uuid_to_pos.get(actor_uuid)
        if actor_pos is None:
            return False, False

        street_bets = [0 for _ in seats] or [0, 0]
        last_raiser = None
        for entry in entries[:index]:
            action = _norm_action(entry.get("action"))
            pos = uuid_to_pos.get(entry.get("uuid"))
            if pos is None or action == "fold":
                continue
            amount = int(entry.get("amount", 0) or 0)
            if action in ("call", "raise", "smallblind", "bigblind"):
                street_bets[pos] = max(street_bets[pos], amount)
            if action == "raise":
                last_raiser = entry.get("uuid")

        own_bet = street_bets[actor_pos] if actor_pos < len(street_bets) else 0
        facing = max(street_bets or [0]) > own_bet
        return facing, facing and last_raiser == our_uuid

    def _current_facing(self, round_state, our_uuid):
        street = round_state.get("street", "preflop")
        histories = round_state.get("action_histories", {}) or {}
        seats = round_state.get("seats", []) or []
        uuid_to_pos = {seat.get("uuid"): i for i, seat in enumerate(seats)}
        our_pos = uuid_to_pos.get(our_uuid)
        if our_pos is None:
            return False
        street_bets = [0 for _ in seats] or [0, 0]
        for entry in histories.get(street, []) or []:
            action = _norm_action(entry.get("action"))
            pos = uuid_to_pos.get(entry.get("uuid"))
            if pos is None or action == "fold":
                continue
            if action in ("call", "raise", "smallblind", "bigblind"):
                street_bets[pos] = max(street_bets[pos], int(entry.get("amount", 0) or 0))
        return max(street_bets or [0]) > street_bets[our_pos]


def _norm_action(action):
    text = (action or "").lower()
    if text in ("smallblind", "bigblind", "ante"):
        return text
    return text


def _normalize(probs):
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 1e-12:
        n = max(1, len(probs))
        return {a: 1.0 / n for a in probs}
    return {a: max(0.0, float(v)) / total for a, v in probs.items()}


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))
