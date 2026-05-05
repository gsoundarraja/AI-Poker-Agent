import argparse
import atexit
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from agent import cfr_abstraction as absn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ABSTRACTION_PATH = os.path.join(DATA_DIR, "cfr_abstraction.json")
STRATEGY_PATH = os.path.join(DATA_DIR, "cfr_strategy.json")
META_PATH = os.path.join(DATA_DIR, "cfr_training_meta.json")
CURVE_PATH = os.path.join(DATA_DIR, "cfr_training_curve.csv")
LOCK_PATH = os.path.join(DATA_DIR, "cfr_training.lock")
TRAINER_RULES_VERSION = "compact_pypokerengine_v2"


BOARD_CARDS_BY_STREET = (0, 3, 4, 5)
DECK_IDS = tuple(range(1, 53))
ACTION_INDEX = {action: idx for idx, action in enumerate(absn.ACTIONS)}


class State:
    __slots__ = (
        "street", "current_player", "street_bets", "contrib",
        "raises_this_street", "total_raises", "acted", "small_blind_player",
        "player_raises_prior", "player_raises_street", "terminal", "utility0"
    )

    def __init__(self, street, current_player, street_bets, contrib,
                 raises_this_street=0, total_raises=0, acted=(False, False),
                 small_blind_player=0, player_raises_prior=(0, 0),
                 player_raises_street=(0, 0),
                 terminal=False, utility0=0.0):
        self.street = street
        self.current_player = current_player
        self.street_bets = street_bets
        self.contrib = contrib
        self.raises_this_street = raises_this_street
        self.total_raises = total_raises
        self.acted = acted
        self.small_blind_player = small_blind_player
        self.player_raises_prior = player_raises_prior
        self.player_raises_street = player_raises_street
        self.terminal = terminal
        self.utility0 = utility0


def initial_state(sb, small_blind_player=0):
    big_blind_player = 1 - small_blind_player
    street_bets = [0, 0]
    street_bets[small_blind_player] = sb
    street_bets[big_blind_player] = 2 * sb
    return State(
        street=0,
        current_player=small_blind_player,
        street_bets=tuple(street_bets),
        contrib=tuple(street_bets),
        acted=(False, False),
        small_blind_player=small_blind_player,
    )


def street_raise_increment(street, sb):
    return 2 * sb if street <= 1 else 4 * sb


def street_raise_limit(street, sb):
    return 4 * street_raise_increment(street, sb)


def legal_actions(state, stack, sb):
    if state.terminal:
        return []
    actions = ["fold", "call"]
    max_bet = max(state.street_bets)
    inc = street_raise_increment(state.street, sb)
    player = state.current_player
    if (
        max_bet < street_raise_limit(state.street, sb)
        and state.player_raises_prior[player] < 4
    ):
        new_amount = max_bet + inc
        if state.contrib[player] + max(0, new_amount - state.street_bets[player]) <= stack:
            actions.append("raise")
    return actions


def apply_action(state, action, deal, stack, sb):
    player = state.current_player
    other = 1 - player
    if action == "fold":
        if player == 0:
            return State(
                state.street, other, state.street_bets, state.contrib,
                raises_this_street=state.raises_this_street,
                total_raises=state.total_raises,
                acted=state.acted,
                small_blind_player=state.small_blind_player,
                player_raises_prior=state.player_raises_prior,
                player_raises_street=state.player_raises_street,
                terminal=True,
                utility0=-float(state.contrib[0]),
            )
        return State(
            state.street, other, state.street_bets, state.contrib,
            raises_this_street=state.raises_this_street,
            total_raises=state.total_raises,
            acted=state.acted,
            small_blind_player=state.small_blind_player,
            player_raises_prior=state.player_raises_prior,
            player_raises_street=state.player_raises_street,
            terminal=True,
            utility0=float(state.contrib[1]),
        )

    street_bets = list(state.street_bets)
    contrib = list(state.contrib)
    acted = list(state.acted)
    max_bet = max(street_bets)

    if action == "raise":
        new_amount = max_bet + street_raise_increment(state.street, sb)
        paid = new_amount - street_bets[player]
        street_bets[player] = new_amount
        contrib[player] += paid
        acted = [False, False]
        acted[player] = True
        player_raises_street = list(state.player_raises_street)
        player_raises_street[player] += 1
        return State(
            street=state.street,
            current_player=other,
            street_bets=tuple(street_bets),
            contrib=tuple(contrib),
            raises_this_street=state.raises_this_street + 1,
            total_raises=state.total_raises + 1,
            acted=tuple(acted),
            small_blind_player=state.small_blind_player,
            player_raises_prior=state.player_raises_prior,
            player_raises_street=tuple(player_raises_street),
        )

    paid = max(0, max_bet - street_bets[player])
    street_bets[player] += paid
    contrib[player] += paid
    acted[player] = True
    if acted[0] and acted[1] and street_bets[0] == street_bets[1]:
        return advance_street(state, tuple(contrib), deal)
    return State(
        street=state.street,
        current_player=other,
        street_bets=tuple(street_bets),
        contrib=tuple(contrib),
        raises_this_street=state.raises_this_street,
        total_raises=state.total_raises,
        acted=tuple(acted),
        small_blind_player=state.small_blind_player,
        player_raises_prior=state.player_raises_prior,
        player_raises_street=state.player_raises_street,
    )


def advance_street(state, contrib, deal):
    if state.street >= 3:
        result = absn.showdown_utility(deal["holes"][0], deal["holes"][1], deal["board"])
        if result > 0:
            util0 = float(contrib[1])
        elif result < 0:
            util0 = -float(contrib[0])
        else:
            util0 = 0.0
        return State(
            state.street, state.small_blind_player, (0, 0), contrib,
            raises_this_street=state.raises_this_street,
            total_raises=state.total_raises,
            acted=state.acted,
            small_blind_player=state.small_blind_player,
            player_raises_prior=state.player_raises_prior,
            player_raises_street=state.player_raises_street,
            terminal=True,
            utility0=util0,
        )
    player_raises_prior = tuple(
        state.player_raises_prior[i] + state.player_raises_street[i]
        for i in range(2)
    )
    return State(
        street=state.street + 1,
        current_player=state.small_blind_player,
        street_bets=(0, 0),
        contrib=contrib,
        raises_this_street=0,
        total_raises=state.total_raises,
        acted=(False, False),
        small_blind_player=state.small_blind_player,
        player_raises_prior=player_raises_prior,
        player_raises_street=(0, 0),
    )


def make_deal(rng):
    deck = rng.sample(DECK_IDS, 9)
    return {
        "holes": (deck[:2], deck[2:4]),
        "board": deck[4:9],
    }


def infoset(metadata, state, deal, sb):
    player = state.current_player
    board_len = BOARD_CARDS_BY_STREET[state.street]
    to_call = max(0, max(state.street_bets) - state.street_bets[player])
    return absn.infoset_key(
        metadata=metadata,
        player=0,
        hole_ids=deal["holes"][player],
        community_ids=deal["board"][:board_len],
        pot=sum(state.contrib),
        to_call=to_call,
        street_bets=state.street_bets,
        raises_this_street=state.raises_this_street,
        total_raises=state.total_raises,
        is_small_blind=(player == state.small_blind_player),
        small_blind=sb,
    )


def regret_matching(regrets, legal):
    legal_idx = [ACTION_INDEX[a] for a in legal]
    positives = [0.0, 0.0, 0.0]
    for i in legal_idx:
        positives[i] = max(0.0, regrets[i])
    total = sum(positives)
    if total <= 1e-12:
        p = 1.0 / len(legal_idx)
        return [p if i in legal_idx else 0.0 for i in range(3)]
    return [positives[i] / total if i in legal_idx else 0.0 for i in range(3)]


def sample_action(strategy, legal, rng):
    pick = rng.random()
    acc = 0.0
    for action in legal:
        idx = ACTION_INDEX[action]
        acc += strategy[idx]
        if pick <= acc:
            return action, max(strategy[idx], 1e-12)
    action = legal[-1]
    return action, max(strategy[ACTION_INDEX[action]], 1e-12)


def add_vec(table, key, idx, value):
    if key not in table:
        table[key] = [0.0, 0.0, 0.0]
    table[key][idx] += value


def get_combined_regrets(base, delta, key, cfr_plus=False):
    b = base.get(key)
    d = delta.get(key)
    if b is None and d is None:
        out = [0.0, 0.0, 0.0]
    elif b is None:
        out = d[:]
    elif d is None:
        out = b[:]
    else:
        out = [b[i] + d[i] for i in range(3)]
    if cfr_plus:
        return [max(0.0, x) for x in out]
    return out


def add_regret(table, base, key, idx, value, cfr_plus=False):
    if key not in table:
        table[key] = [0.0, 0.0, 0.0]
    if not cfr_plus:
        table[key][idx] += value
        return
    base_vec = base.get(key)
    base_value = base_vec[idx] if base_vec is not None else 0.0
    current = max(0.0, base_value + table[key][idx])
    table[key][idx] = max(0.0, current + value) - base_value


def cfr(state, deal, traverser, metadata, base_regrets, regret_delta, strategy_delta,
        legal_masks, rng, stack, sb, reach_i, reach_opp, cfr_plus=False):
    if state.terminal:
        return state.utility0 if traverser == 0 else -state.utility0

    legal = legal_actions(state, stack, sb)
    key = infoset(metadata, state, deal, sb)
    mask = legal_masks.setdefault(key, [0, 0, 0])
    for action in legal:
        mask[ACTION_INDEX[action]] = 1
    regrets = get_combined_regrets(base_regrets, regret_delta, key, cfr_plus)
    strategy = regret_matching(regrets, legal)
    player = state.current_player

    if player == traverser:
        action_utils = {}
        node_util = 0.0
        for action in legal:
            idx = ACTION_INDEX[action]
            util = cfr(
                apply_action(state, action, deal, stack, sb), deal, traverser,
                metadata, base_regrets, regret_delta, strategy_delta, legal_masks,
                rng, stack, sb, reach_i * strategy[idx], reach_opp, cfr_plus,
            )
            action_utils[action] = util
            node_util += strategy[idx] * util
        for action in legal:
            idx = ACTION_INDEX[action]
            add_regret(
                regret_delta, base_regrets, key, idx,
                reach_opp * (action_utils[action] - node_util), cfr_plus
            )
            add_vec(strategy_delta, key, idx, reach_i * strategy[idx])
        return node_util

    action, prob = sample_action(strategy, legal, rng)
    return cfr(
        apply_action(state, action, deal, stack, sb), deal, traverser,
        metadata, base_regrets, regret_delta, strategy_delta, legal_masks,
        rng, stack, sb, reach_i, reach_opp * prob, cfr_plus,
    )


def worker_chunk(args):
    (worker_id, seed, iterations, stack, sb, metadata, base_regrets, cfr_plus) = args
    rng = random.Random(seed + 7919 * worker_id)
    regret_delta = {}
    strategy_delta = {}
    legal_masks = {}
    util_sum = 0.0
    for _ in range(iterations):
        deal = make_deal(rng)
        small_blind_player = rng.randrange(2)
        util_sum += cfr(initial_state(sb, small_blind_player), deal, 0, metadata, base_regrets, regret_delta,
                        strategy_delta, legal_masks, rng, stack, sb, 1.0, 1.0, cfr_plus)
        util_sum += cfr(initial_state(sb, small_blind_player), deal, 1, metadata, base_regrets, regret_delta,
                        strategy_delta, legal_masks, rng, stack, sb, 1.0, 1.0, cfr_plus)
    return {
        "regret_delta": regret_delta,
        "strategy_delta": strategy_delta,
        "legal_masks": legal_masks,
        "traversals": iterations * 2,
        "utility_sum": util_sum,
    }


def merge_vecs(dst, src, cfr_plus=False):
    for key, vec in src.items():
        if key not in dst:
            dst[key] = [0.0, 0.0, 0.0]
        cur = dst[key]
        for i in range(3):
            cur[i] = cur[i] + vec[i]
            if cfr_plus:
                cur[i] = max(0.0, cur[i])


def merge_masks(dst, src):
    for key, vec in src.items():
        if key not in dst:
            dst[key] = [0, 0, 0]
        cur = dst[key]
        for i in range(3):
            cur[i] = 1 if cur[i] or vec[i] else 0


def average_strategy(strategy_sum, regret_sum, legal_masks):
    out = {}
    keys = set(strategy_sum) | set(regret_sum) | set(legal_masks)
    for key in keys:
        mask = legal_masks.get(key, [1, 1, 1])
        legal_idx = [i for i, ok in enumerate(mask) if ok]
        if not legal_idx:
            legal_idx = [0, 1, 2]
        strat = strategy_sum.get(key, [0.0, 0.0, 0.0])
        total = sum(max(0.0, strat[i]) for i in legal_idx)
        if total <= 1e-12:
            probs = regret_matching(regret_sum.get(key, [0.0, 0.0, 0.0]),
                                    [absn.ACTIONS[i] for i in legal_idx])
        else:
            probs = [0.0, 0.0, 0.0]
            for i in legal_idx:
                probs[i] = max(0.0, strat[i]) / total
        out[key] = {absn.ACTIONS[i]: probs[i] for i in legal_idx}
    return out


def save_checkpoint(strategy_path, meta_path, abstraction_path, metadata, regret_sum,
                    strategy_sum, legal_masks, meta):
    avg = average_strategy(strategy_sum, regret_sum, legal_masks)
    payload = {
        "version": 1,
        "action_order": list(absn.ACTIONS),
        "regret_update": meta.get("regret_update", "cfr"),
        "average_strategy": avg,
        "regret_sum": regret_sum,
        "strategy_sum": strategy_sum,
        "legal_masks": legal_masks,
    }
    tmp = strategy_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, strategy_path)
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp_meta, meta_path)
    absn.save_abstraction(abstraction_path, metadata)


def append_curve_row(path, row):
    fieldnames = [
        "run_id",
        "timestamp",
        "previous_batches",
        "previous_traversals",
        "previous_elapsed_sec",
        "run_batches",
        "run_traversals",
        "run_elapsed_sec",
        "total_batches",
        "total_traversals",
        "total_elapsed_sec",
        "batches",
        "traversals",
        "elapsed_sec",
        "workers",
        "merge_interval",
        "regret_update",
        "stack",
        "small_blind",
        "infosets",
        "utility_sum_last_batch",
        "checkpoint_reason",
    ]
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != fieldnames:
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for old_row in old_rows:
                    migrated = dict(old_row)
                    migrated.setdefault("run_batches", old_row.get("batches", ""))
                    migrated.setdefault("run_traversals", old_row.get("traversals", ""))
                    migrated.setdefault("run_elapsed_sec", old_row.get("elapsed_sec", ""))
                    migrated.setdefault("total_batches", old_row.get("batches", ""))
                    migrated.setdefault("total_traversals", old_row.get("traversals", ""))
                    migrated.setdefault("total_elapsed_sec", old_row.get("elapsed_sec", ""))
                    migrated.setdefault("regret_update", old_row.get("regret_update", "cfr"))
                    writer.writerow({name: migrated.get(name, "") for name in fieldnames})
            os.replace(tmp, path)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_meta(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}, {}, {}
    with open(path, "r") as f:
        payload = json.load(f)
    return (
        payload.get("regret_sum", {}),
        payload.get("strategy_sum", {}),
        payload.get("legal_masks", {}),
    )


def acquire_training_lock(path, force=False):
    if force and os.path.exists(path):
        os.remove(path)
    payload = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Remove this file only if no train_mccfr.py process is running.",
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            with open(path, "r") as f:
                existing = f.read().strip()
        except Exception:
            existing = ""
        raise RuntimeError(
            "Refusing to train because {} already exists. Another trainer may "
            "be writing cfr_strategy.json. Stop that process, or rerun with "
            "--force-lock only after verifying the lock is stale. Lock contents: {}".format(
                path, existing or "<unreadable>"
            )
        )
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    def cleanup():
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    atexit.register(cleanup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--merge-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-interval-sec", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=683)
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=10)
    parser.add_argument("--abstraction-samples", type=int, default=20000)
    parser.add_argument("--card-buckets", type=int, default=8)
    parser.add_argument("--pot-buckets", type=int, default=6)
    parser.add_argument("--to-call-buckets", type=int, default=5)
    parser.add_argument("--rebuild-abstraction", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--target-total-traversals",
        type=int,
        default=0,
        help="Stop after the checkpoint reaches at least this total traversal count",
    )
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument("--cfr-plus", action="store_true")
    args = parser.parse_args()
    regret_update = "cfr_plus" if args.cfr_plus else "cfr"

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        acquire_training_lock(LOCK_PATH, force=args.force_lock)
    except RuntimeError as exc:
        sys.exit(str(exc))

    if args.rebuild_abstraction or not os.path.exists(ABSTRACTION_PATH):
        print("Learning abstraction from {} sampled deals/traces...".format(args.abstraction_samples), flush=True)
        metadata = absn.learn_abstraction(
            args.abstraction_samples,
            args.seed,
            card_buckets=args.card_buckets,
            pot_buckets=args.pot_buckets,
            to_call_buckets=args.to_call_buckets,
        )
        absn.save_abstraction(ABSTRACTION_PATH, metadata)
    else:
        metadata = absn.load_abstraction(ABSTRACTION_PATH)

    previous_meta = load_meta(META_PATH) if args.resume else {}
    previous_batches = int(previous_meta.get("total_batches", previous_meta.get("batches", 0)) or 0)
    previous_traversals = int(previous_meta.get("total_traversals", previous_meta.get("traversals", 0)) or 0)
    previous_elapsed = float(previous_meta.get("total_elapsed_sec", previous_meta.get("elapsed_sec", 0.0)) or 0.0)

    if args.resume:
        previous_rules = previous_meta.get("trainer_rules")
        previous_regret_update = previous_meta.get("regret_update", "cfr")
        if previous_rules and previous_rules != TRAINER_RULES_VERSION:
            print(
                "WARNING: checkpoint trainer_rules={} differs from current {}. "
                "For a clean rule-matched policy, train without --resume.".format(
                    previous_rules, TRAINER_RULES_VERSION
                ),
                flush=True,
            )
        elif not previous_rules and previous_meta:
            print(
                "WARNING: checkpoint has no trainer_rules metadata. "
                "If it was trained before rule-matching, --resume will mix training games.",
                flush=True,
            )
        if previous_regret_update != regret_update:
            sys.exit(
                "Refusing to resume {} checkpoint with {} mode. Train from scratch "
                "without --resume, or use the matching regret mode.".format(
                    previous_regret_update, regret_update
                )
            )
        regret_sum, strategy_sum, legal_masks = load_checkpoint(STRATEGY_PATH)
        print(
            "Resumed {} regret infosets; previous total traversals={}".format(
                len(regret_sum), previous_traversals
            ),
            flush=True,
        )
    else:
        regret_sum, strategy_sum, legal_masks = {}, {}, {}

    start = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    deadline = start + max(0.0, args.minutes) * 60.0
    next_checkpoint = start + args.checkpoint_interval_sec
    traversals = 0
    batches = 0
    workers = max(1, args.workers)
    print("Training MCCFR: workers={}, merge_interval={}, stack={}, sb={}".format(
        workers, args.merge_interval, args.stack, args.small_blind), flush=True)
    print("Regret update: {}".format(regret_update), flush=True)
    if args.target_total_traversals:
        print("Target total traversals: {}".format(args.target_total_traversals), flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            if args.max_batches and batches >= args.max_batches:
                break
            if args.target_total_traversals and previous_traversals + traversals >= args.target_total_traversals:
                break
            if args.minutes > 0 and time.time() >= deadline:
                break
            snapshot = {k: v[:] for k, v in regret_sum.items()}
            futures = [
                pool.submit(worker_chunk, (
                    wid, args.seed + batches * workers + wid, args.merge_interval,
                    args.stack, args.small_blind, metadata, snapshot, args.cfr_plus,
                ))
                for wid in range(workers)
            ]
            util = 0.0
            for fut in as_completed(futures):
                result = fut.result()
                merge_vecs(regret_sum, result["regret_delta"], cfr_plus=args.cfr_plus)
                merge_vecs(strategy_sum, result["strategy_delta"])
                merge_masks(legal_masks, result["legal_masks"])
                traversals += result["traversals"]
                util += result["utility_sum"]
            batches += 1
            elapsed = time.time() - start
            total_batches = previous_batches + batches
            total_traversals = previous_traversals + traversals
            total_elapsed = previous_elapsed + elapsed
            print(
                "batch={} run_traversals={} total_traversals={} infosets={} elapsed={:.1f}s util_sum={:.1f}".format(
                    batches, traversals, total_traversals, len(regret_sum), elapsed, util
                ),
                flush=True,
            )
            if time.time() >= next_checkpoint:
                meta = {
                    "run_id": run_id,
                    "previous_batches": previous_batches,
                    "previous_traversals": previous_traversals,
                    "previous_elapsed_sec": previous_elapsed,
                    "run_batches": batches,
                    "run_traversals": traversals,
                    "run_elapsed_sec": elapsed,
                    "total_batches": total_batches,
                    "total_traversals": total_traversals,
                    "total_elapsed_sec": total_elapsed,
                    "batches": total_batches,
                    "traversals": total_traversals,
                    "elapsed_sec": total_elapsed,
                    "workers": workers,
                    "merge_interval": args.merge_interval,
                    "stack": args.stack,
                    "small_blind": args.small_blind,
                    "seed": args.seed,
                    "trainer_rules": TRAINER_RULES_VERSION,
                    "regret_update": regret_update,
                    "infosets": len(regret_sum),
                    "utility_sum_last_batch": util,
                }
                save_checkpoint(
                    STRATEGY_PATH, META_PATH, ABSTRACTION_PATH, metadata,
                    regret_sum, strategy_sum, legal_masks,
                    meta,
                )
                curve_row = dict(meta)
                curve_row["timestamp"] = time.time()
                curve_row["checkpoint_reason"] = "interval"
                append_curve_row(CURVE_PATH, curve_row)
                next_checkpoint = time.time() + args.checkpoint_interval_sec

    elapsed = time.time() - start
    total_batches = previous_batches + batches
    total_traversals = previous_traversals + traversals
    total_elapsed = previous_elapsed + elapsed
    meta = {
        "run_id": run_id,
        "previous_batches": previous_batches,
        "previous_traversals": previous_traversals,
        "previous_elapsed_sec": previous_elapsed,
        "run_batches": batches,
        "run_traversals": traversals,
        "run_elapsed_sec": elapsed,
        "total_batches": total_batches,
        "total_traversals": total_traversals,
        "total_elapsed_sec": total_elapsed,
        "batches": total_batches,
        "traversals": total_traversals,
        "elapsed_sec": total_elapsed,
        "workers": workers,
        "merge_interval": args.merge_interval,
        "stack": args.stack,
        "small_blind": args.small_blind,
        "seed": args.seed,
        "trainer_rules": TRAINER_RULES_VERSION,
        "regret_update": regret_update,
        "infosets": len(regret_sum),
        "utility_sum_last_batch": util if batches else 0.0,
    }
    save_checkpoint(
        STRATEGY_PATH, META_PATH, ABSTRACTION_PATH, metadata,
        regret_sum, strategy_sum, legal_masks,
        meta,
    )
    curve_row = dict(meta)
    curve_row["timestamp"] = time.time()
    curve_row["checkpoint_reason"] = "final"
    append_curve_row(CURVE_PATH, curve_row)
    print("Wrote {}".format(STRATEGY_PATH), flush=True)
    print("Wrote {}".format(META_PATH), flush=True)
    print("Appended {}".format(CURVE_PATH), flush=True)


if __name__ == "__main__":
    main()
