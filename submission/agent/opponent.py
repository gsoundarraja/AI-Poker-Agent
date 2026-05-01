from collections import defaultdict

class OpponentModel:
    def __init__(self):
        self.stats = defaultdict(self._empty_stats)

    @staticmethod
    def _empty_stats():
        return {
            "per_street": defaultdict(lambda: {"fold": 0.0, "call": 0.0, "raise": 0.0}),
            "per_street_facing_raise": defaultdict(lambda: {"fold": 0.0, "call": 0.0, "raise": 0.0})
        }
    # record op action
    def update_with_action(self, opp_uuid, street, action, facing_raise):
        if action not in ("fold", "call", "raise"):
            return
        s = self.stats[opp_uuid]
        s["per_street"][street][action] +=1.0
        # mark if it was raise
        if facing_raise:
            s["per_street_facing_raise"][street][action] +=1.0

    # predict opp policy
    def predict_action_distribution(self, opp_uuid, street, facing_raise = False):
        s = self.stats.get(opp_uuid)
        # first round prior
        if s is None:
            return {"fold": 0.25, "call": 0.55, "raise": 0.20}
        table = s["per_street_facing_raise"] if facing_raise else s["per_street"]
        counts = table.get(street)
        if counts is None or sum(counts.values()) == 0:
            counts = {"fold": 0.0, "call": 0.0, "raise": 0.0}
            for street_counts in s["per_street"].values():
                for a, c in street_counts.items():
                    counts[a] += c
        # laplace smoothing +2
        smoothed = {a: counts.get(a, 0.0) + 2 for a in ("fold", "call", "raise")}
        total = sum(smoothed.values())
        return {a: smoothed[a] / total for a in smoothed}

    # Pr(opp folds | raise)
    def fold_equity(self, opp_uuid, street):
        dist = self.predict_action_distribution(opp_uuid, street, facing_raise =True)
        return dist["fold"]