import random

from pypokerengine.players import BasePokerPlayer

from reporting.gather_baselines import _hand_strength, _is_facing_bet


class ParametricStyleOpponent(BasePokerPlayer):
    def __init__(
        self,
        seed=0,
        raise_hi=0.78,
        call_lo=0.42,
        fold_lo=0.38,
        bluff=0.04,
        preflop_raise=0.18,
        pressure_raise=0.10,
        sticky_call=0.70,
        trap_slowplay=0.0,
    ):
        super().__init__()
        self._rng = random.Random(seed)
        self.raise_hi = float(raise_hi)
        self.call_lo = float(call_lo)
        self.fold_lo = float(fold_lo)
        self.bluff = float(bluff)
        self.preflop_raise = float(preflop_raise)
        self.pressure_raise = float(pressure_raise)
        self.sticky_call = float(sticky_call)
        self.trap_slowplay = float(trap_slowplay)

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = {a["action"] for a in valid_actions}
        street = round_state.get("street", "preflop")
        facing = _is_facing_bet(round_state, getattr(self, "uuid", None))
        strength = _hand_strength(hole_card, round_state)
        late = street in ("turn", "river")

        if "raise" in legal:
            raise_cutoff = self.raise_hi - (0.04 if late else 0.0)
            if strength >= raise_cutoff and self._rng.random() >= self.trap_slowplay:
                return "raise"
            if street == "preflop" and self._rng.random() < self.preflop_raise:
                return "raise"
            if not facing and self._rng.random() < self.bluff:
                return "raise"
            if facing and self._rng.random() < self.pressure_raise:
                return "raise"

        if facing:
            if strength < self.fold_lo and "fold" in legal:
                if self._rng.random() > self.sticky_call:
                    return "fold"
            if "call" in legal and (strength >= self.call_lo or self._rng.random() < self.sticky_call):
                return "call"
            if "fold" in legal:
                return "fold"

        if "call" in legal:
            return "call"
        return next(iter(legal))

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


FAMILIES = ("value", "nit", "pressure", "station", "loose_aggressive", "folder")
FAMILY_INDEX = {name: i for i, name in enumerate(FAMILIES)}


def family_params(family, seed):
    rng = random.Random((FAMILY_INDEX[family] + 1) * 1_000_003 + int(seed))
    if family == "value":
        return {
            "seed": seed,
            "raise_hi": rng.uniform(0.70, 0.84),
            "call_lo": rng.uniform(0.36, 0.50),
            "fold_lo": rng.uniform(0.32, 0.46),
            "bluff": rng.uniform(0.00, 0.06),
            "preflop_raise": rng.uniform(0.08, 0.24),
            "pressure_raise": rng.uniform(0.02, 0.12),
            "sticky_call": rng.uniform(0.25, 0.55),
            "trap_slowplay": rng.uniform(0.00, 0.08),
        }
    if family == "nit":
        return {
            "seed": seed,
            "raise_hi": rng.uniform(0.82, 0.94),
            "call_lo": rng.uniform(0.48, 0.62),
            "fold_lo": rng.uniform(0.48, 0.66),
            "bluff": rng.uniform(0.00, 0.02),
            "preflop_raise": rng.uniform(0.02, 0.12),
            "pressure_raise": rng.uniform(0.00, 0.05),
            "sticky_call": rng.uniform(0.05, 0.28),
            "trap_slowplay": rng.uniform(0.00, 0.10),
        }
    if family == "pressure":
        return {
            "seed": seed,
            "raise_hi": rng.uniform(0.62, 0.78),
            "call_lo": rng.uniform(0.32, 0.48),
            "fold_lo": rng.uniform(0.28, 0.44),
            "bluff": rng.uniform(0.08, 0.20),
            "preflop_raise": rng.uniform(0.18, 0.42),
            "pressure_raise": rng.uniform(0.12, 0.34),
            "sticky_call": rng.uniform(0.20, 0.48),
            "trap_slowplay": rng.uniform(0.00, 0.05),
        }
    if family == "station":
        return {
            "seed": seed,
            "raise_hi": rng.uniform(0.76, 0.90),
            "call_lo": rng.uniform(0.20, 0.40),
            "fold_lo": rng.uniform(0.18, 0.34),
            "bluff": rng.uniform(0.00, 0.04),
            "preflop_raise": rng.uniform(0.03, 0.15),
            "pressure_raise": rng.uniform(0.00, 0.08),
            "sticky_call": rng.uniform(0.70, 0.98),
            "trap_slowplay": rng.uniform(0.00, 0.12),
        }
    if family == "loose_aggressive":
        return {
            "seed": seed,
            "raise_hi": rng.uniform(0.55, 0.72),
            "call_lo": rng.uniform(0.26, 0.42),
            "fold_lo": rng.uniform(0.22, 0.38),
            "bluff": rng.uniform(0.12, 0.30),
            "preflop_raise": rng.uniform(0.35, 0.75),
            "pressure_raise": rng.uniform(0.20, 0.45),
            "sticky_call": rng.uniform(0.35, 0.70),
            "trap_slowplay": rng.uniform(0.00, 0.04),
        }
    if family == "folder":
        return {
            "seed": seed,
            "raise_hi": rng.uniform(0.78, 0.92),
            "call_lo": rng.uniform(0.45, 0.62),
            "fold_lo": rng.uniform(0.54, 0.76),
            "bluff": rng.uniform(0.00, 0.06),
            "preflop_raise": rng.uniform(0.04, 0.18),
            "pressure_raise": rng.uniform(0.00, 0.06),
            "sticky_call": rng.uniform(0.00, 0.18),
            "trap_slowplay": rng.uniform(0.00, 0.06),
        }
    raise ValueError("unknown randomized opponent family: {}".format(family))


def randomized_opponent_specs(split, per_family=2, seed=683):
    split_offsets = {"train": 10_000, "validation": 20_000, "test": 30_000}
    offset = split_offsets.get(split, 40_000)
    specs = []
    for family in FAMILIES:
        for i in range(per_family):
            opp_seed = offset + seed + 101 * i + 997 * FAMILIES.index(family)
            name = "{}_{}_{}".format(split, family, i)
            specs.append((name, ParametricStyleOpponent, family_params(family, opp_seed)))
    return specs
