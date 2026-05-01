import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pokeragent import PokerAgent
from training.selfplay import run_match
from reporting.variants import VARIANTS


OUT_CSV = os.path.join(ROOT, "final_project", "tables", "ablations.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--stack", type=int, default=10000)
    ap.add_argument("--small-blind", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    t0 = time.time()
    n_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        with ThreadPoolExecutor(max_workers = len(VARIANTS)) as tpool:
            futures = {}
            for name, VariantCls in VARIANTS:
                print("Submitting Full vs {}".format(name))
                futures[name] = tpool.submit(run_match, PokerAgent, VariantCls, "Full", name,
                                             args.games, args.hands, args.stack, args.small_blind, 0, pool, name)
            with open(OUT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["variant", "games", "hands_total",
                            "full_chips", "variant_chips",
                            "full_wins", "variant_wins",
                            "chips_per_game_full", "elapsed_sec"])
                for name, fut in futures.items():
                    res = fut.result()
                    w.writerow([name, res["games"], res["hands_total"],
                                res["agent1_chips"], res["agent2_chips"],
                                res["wins1"], res["wins2"],
                                round(res["chips_per_game_1"], 1),
                                round(res["elapsed_sec"], 1)])
                    f.flush()
                    print("  Full vs {}: {:+.1f}  ({}/{} games)"
                          .format(name, res["chips_per_game_1"], res["wins1"], res["games"]))
    print("Wrote {}  (total {:.1f}s)".format(OUT_CSV, time.time() - t0))


if __name__ == "__main__":
    main()
