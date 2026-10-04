#!/usr/bin/env python
"""Benchmark the engine across the real 65-unit roster and compute the ranking statistic.

Three numbers matter and they are different:

  * per-unit events/sec on the SIMULATION LOOP -- comparable to the baseline's self-reported
    events.json figure and to the Development clip of 1e7/unit.
  * the ARITHMETIC MEAN of per-unit rates -- the actual leaderboard statistic
    (LEADERBOARD_SORT = "desc" over the complete roster), not a geometric mean or a median.
  * the modelled FINAL score, mean_i [ n_i / (T + n_i/R_i) ], where T is container start-up
    (measured at 92.45 ms for a static binary). Final timing is whole-container wall clock, so
    for small units T dominates no matter how fast the engine is.

    python bench_roster.py ./run_unit [--repeats 5] [--no-outputs]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

def _find_repo() -> "Path":
    """Locate the track3-simulation-public checkout, wherever engine/ happens to live.

    Honours $T3_REPO, then searches upward for a directory containing `units/`. This replaces a
    hardcoded relative path that only worked for one directory layout.
    """
    import os
    from pathlib import Path as _P
    env = os.environ.get("T3_REPO")
    if env and (_P(env) / "units").is_dir():
        return _P(env)
    here = _P(__file__).resolve().parent
    for base in [here, *here.parents]:
        if (base / "units").is_dir():
            return base
        cand = base / "track3-simulation-public"
        if (cand / "units").is_dir():
            return cand
    return _P("../../../track3-simulation-public")
T_CONTAINER = 0.09245  # measured: scratch + static binary, 2x30 repeats, Codespace
BASELINE_MEAN = 14159.0  # arithmetic mean of the 65 shipped events.json rates


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-outputs", action="store_true")
    args = ap.parse_args(argv[1:])

    units = sorted(p.parent for p in REPO.glob("units/*/scenario.json"))
    extra = ["--no-outputs"] if args.no_outputs else []

    rows = []
    for u in units:
        cmd = [args.binary, str(u / "scenario.json"), "--bench", str(args.repeats), *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"{u.name}: ERROR {p.stderr.strip()[:120]}", file=sys.stderr)
            continue
        n_trace, n_msgs, best_ms, median_ms, parse_ms = p.stdout.split()
        n_trace, n_msgs = int(n_trace), int(n_msgs)
        best, median, parse = float(best_ms), float(median_ms), float(parse_ms)
        ref = json.loads((u / "events.json").read_text())
        rows.append({
            "unit": u.name, "n": n_trace, "msgs": n_msgs,
            "best_ms": best, "median_ms": median, "parse_ms": parse,
            "eps": n_trace / (best / 1000.0),
            "ref_eps": float(ref["events_per_sec"]),
        })

    rows.sort(key=lambda r: r["n"])
    print(f"outputs={'off' if args.no_outputs else 'trace+ledger'}  repeats={args.repeats}\n")
    print(f"{'unit':38} {'events':>9} {'best_ms':>9} {'events/sec':>12} {'baseline':>10} {'x base':>8}")
    print("-" * 92)
    for r in rows[:6] + [None] + rows[-6:]:
        if r is None:
            print(f"{'  ... ' + str(len(rows) - 12) + ' more ...':38}")
            continue
        print(f"{r['unit'][:37]:38} {r['n']:>9,} {r['best_ms']:>9.2f} {r['eps']:>12,.0f} "
              f"{r['ref_eps']:>10,.0f} {r['eps'] / r['ref_eps']:>7.0f}x")

    eps = [r["eps"] for r in rows]
    ns = [r["n"] for r in rows]
    arith = st.mean(eps)
    print(f"\n--- the ranking statistic (arithmetic mean of per-unit rates) ---")
    print(f"  engine      : {arith:>14,.0f} events/sec")
    print(f"  baseline    : {BASELINE_MEAN:>14,.0f}   -> {arith / BASELINE_MEAN:.0f}x faster")
    print(f"  Dev clip    : {1e7:>14,.0f}   -> {'SATURATED' if arith >= 1e7 else f'{1e7 / arith:.1f}x short'}")
    print(f"  median unit : {st.median(eps):>14,.0f}")
    print(f"  slowest unit: {min(eps):>14,.0f}  ({min(rows, key=lambda r: r['eps'])['unit']})")
    print(f"  fastest unit: {max(eps):>14,.0f}  ({max(rows, key=lambda r: r['eps'])['unit']})")

    total_ms = sum(r["best_ms"] for r in rows)
    print(f"\n  whole roster: {sum(ns):,} events in {total_ms:,.0f} ms of simulation")
    print(f"  JSON parse  : {sum(r['parse_ms'] for r in rows):,.1f} ms total "
          f"({st.mean([r['parse_ms'] for r in rows]):.2f} ms/unit)")

    print(f"\n--- modelled FINAL score (container wall clock, T={T_CONTAINER * 1000:.1f} ms) ---")
    # Per unit the engine's own rate differs, so use each unit's measured rate.
    final = st.mean([
        r["n"] / (T_CONTAINER + r["n"] / r["eps"]) for r in rows
    ])
    ceiling = st.mean([r["n"] / T_CONTAINER for r in rows])
    print(f"  with measured per-unit rates : {final:>12,.0f}")
    print(f"  ceiling if the engine were infinitely fast : {ceiling:>12,.0f}")
    print(f"  -> the engine is already at {final / ceiling * 100:.0f}% of what container "
          f"start-up permits")
    print(f"  -> all remaining engine optimisation is worth at most "
          f"{ceiling / final:.2f}x on the Final board")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
