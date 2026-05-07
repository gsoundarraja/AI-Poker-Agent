from collections import defaultdict

from pypokerengine.engine.card import Card

from . import cfr_abstraction as absn


class PreflopLookup:
    def __init__(self, table=None, abstraction=None):
        self.table = table or {}
        self.abstraction = abstraction or absn.default_abstraction()

    @classmethod
    def from_policy(cls, strategy, abstraction):
        table = {}
        buckets = abstraction.get("preflop_bucket_map", {})
        by_bucket_context = defaultdict(list)

        for infoset, probs in (strategy or {}).items():
            vals = absn.parse_infoset_key(infoset)
            if vals is None or vals[absn.INFOSET_STREET] != 0:
                continue
            card_bucket = vals[absn.INFOSET_CARD_BUCKET]
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
            runtime_vals = absn.parse_infoset_key(
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
            return None
        return absn.mask_probs(probs, valid_actions)


def _context_from_vals(vals):
    return "|".join(str(x) for x in (
        vals[absn.INFOSET_POSITION],
        vals[absn.INFOSET_POT_BUCKET],
        vals[absn.INFOSET_CALL_BUCKET],
        vals[absn.INFOSET_STREET_RAISES],
        vals[absn.INFOSET_TOTAL_RAISES],
        vals[absn.INFOSET_MAX_BET],
        vals[absn.INFOSET_FACING],
        vals[absn.INFOSET_PLAYER],
    ))


def _table_key(klass, context):
    return "{}:{}".format(klass, context)


def _average_probs(prob_list):
    totals = {a: 0.0 for a in absn.ACTIONS}
    for probs in prob_list:
        for action in absn.ACTIONS:
            totals[action] += max(0.0, float(probs.get(action, 0.0)))
    n = float(max(1, len(prob_list)))
    return absn.normalize_probs({a: v / n for a, v in totals.items()})
