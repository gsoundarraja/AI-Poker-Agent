from collections import defaultdict

from pypokerengine.engine.card import Card

from . import cfr_abstraction as absn


class PreflopLookup:
    def __init__(self, table=None, abstraction=None):
        self.table = table or {}
        self.abstraction = abstraction or absn.default_abstraction()
        self._class_index = defaultdict(dict)
        for key, probs in self.table.items():
            klass, context = key.split(":", 1)
            self._class_index[klass][context] = probs

    @classmethod
    def from_policy(cls, strategy, abstraction):
        table = {}
        buckets = abstraction.get("preflop_bucket_map", {})
        by_bucket_context = defaultdict(list)

        for infoset, probs in (strategy or {}).items():
            vals = _parse_infoset(infoset)
            if vals is None or vals[0] != 0:
                continue
            card_bucket = vals[3]
            context = _context_from_vals(vals)
            by_bucket_context[(card_bucket, context)].append(probs)

        for klass in absn.all_preflop_classes():
            card_bucket = int(buckets.get(klass, 0))
            for (bucket, context), prob_list in by_bucket_context.items():
                if bucket != card_bucket:
                    continue
                table[_table_key(klass, context)] = _average_probs(prob_list)
        return cls(table, abstraction=abstraction)

    def distribution(self, valid_actions, hole_card, round_state, player_uuid):
        try:
            if round_state.get("street", "preflop") != "preflop":
                return None
            hole_ids = [Card.from_str(c).to_id() for c in hole_card]
            klass = absn.preflop_class_from_ids(hole_ids)
            runtime_vals = _parse_infoset(
                absn.runtime_infoset(
                    self.abstraction,
                    round_state,
                    hole_card,
                    player_uuid,
                )
            )
        except Exception:
            return None
        if runtime_vals is None:
            return None

        context = _context_from_vals(runtime_vals)
        probs = self.table.get(_table_key(klass, context))
        if probs is None:
            probs = self._nearest_for_class(klass, context)
        if probs is None:
            return None
        return _mask_and_normalize(probs, valid_actions)

    def _nearest_for_class(self, klass, context):
        options = self._class_index.get(klass)
        if not options:
            return None
        target = tuple(int(x) for x in context.split("|"))
        best_context = None
        best_dist = None
        for cand in options:
            vals = tuple(int(x) for x in cand.split("|"))
            dist = (
                2 * abs(vals[0] - target[0])
                + 2 * abs(vals[1] - target[1])
                + abs(vals[2] - target[2])
                + abs(vals[3] - target[3])
                + abs(vals[4] - target[4])
                + abs(vals[5] - target[5])
                + abs(vals[6] - target[6])
                + 4 * abs(vals[7] - target[7])
            )
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_context = cand
        return options.get(best_context)


def _parse_infoset(key):
    try:
        vals = tuple(int(x) for x in key.split("|"))
    except Exception:
        return None
    return vals if len(vals) == 10 else None


def _context_from_vals(vals):
    return "|".join(str(x) for x in (
        vals[2],
        vals[4],
        vals[5],
        vals[6],
        vals[7],
        vals[8],
        vals[9],
        vals[1],
    ))


def _table_key(klass, context):
    return "{}:{}".format(klass, context)


def _average_probs(prob_list):
    totals = {a: 0.0 for a in absn.ACTIONS}
    for probs in prob_list:
        for action in absn.ACTIONS:
            totals[action] += max(0.0, float(probs.get(action, 0.0)))
    n = float(max(1, len(prob_list)))
    return _normalize({a: v / n for a, v in totals.items()})


def _mask_and_normalize(probs, valid_actions):
    legal = [a["action"] for a in valid_actions]
    call_amount = None
    for action in valid_actions:
        if action.get("action") == "call":
            call_amount = action.get("amount")
            break
    if isinstance(call_amount, (int, float)) and call_amount <= 0 and "call" in legal:
        legal = [a for a in legal if a != "fold"]
    return _normalize({a: max(0.0, float(probs.get(a, 0.0))) for a in legal})


def _normalize(probs):
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 1e-12:
        n = max(1, len(probs))
        return {a: 1.0 / n for a in probs}
    return {a: max(0.0, float(v)) / total for a, v in probs.items()}
