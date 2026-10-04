#!/usr/bin/env python
"""Compare the C++ oracle against the REAL patched ABIDES SparseMeanRevertingOracle.

This is not a comparison against my reading of the algorithm -- it imports the actual module
from the pinned checkout (f9cbe51 + oracle_scheduled_jump.patch) and drives both
implementations from identical seeds. `abides_core` is stubbed down to `NanosecondTime = int`
because its real __init__ pulls in the kernel; pandas and the type alias are the only things the
oracle module needs from it.

    python check_oracle.py ./test_rng [--abides C:/path/to/abides-src]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

DATE_NS = 1612483200000000000      # pd.to_datetime("20210205").value
MKT_OPEN = DATE_NS + 34200000000000  # + 09:30:00


def load_real_oracle(abides_root: Path):
    stub = types.ModuleType("abides_core")
    stub.NanosecondTime = int
    sys.modules["abides_core"] = stub
    sys.path.insert(0, str(abides_root / "abides-markets"))
    from abides_markets.oracles.sparse_mean_reverting_oracle import (  # noqa: E402
        SparseMeanRevertingOracle,
    )
    return SparseMeanRevertingOracle


def python_obs(Oracle, c: dict) -> list[int]:
    """Drive the real oracle exactly as config.py would."""
    np.random.seed(c["global_seed"])
    symbol_rs = np.random.RandomState(np.random.randint(0, 2**32, dtype="uint64"))
    mkt_close = MKT_OPEN + c["horizon_ns"]
    jumps = []
    if c["jump_time_ns"] >= 0:
        jumps.append({
            "time_ns": MKT_OPEN + c["jump_time_ns"],
            "magnitude": int(c["jump_mag"]),
            "_consumed": False,
        })
    symbols = {
        "ABM": {
            "r_bar": int(c["r_bar"]),
            "kappa": c["kappa_per_s"] / 1e9 if c["kappa_per_s"] > 0 else 1.67e-16,
            "sigma_s": 0,
            "fund_vol": c["sigma"],
            "megashock_lambda_a": (c["jump_intensity"] / 1e9
                                   if c["jump_intensity"] > 0 else 2.77778e-18),
            "megashock_mean": float(c["jump_sigma"]) or 1000.0,
            "megashock_var": 50_000,
            "random_state": symbol_rs,
            "scheduled_jumps": jumps,
        }
    }
    oracle = Oracle(MKT_OPEN, mkt_close, symbols)
    agent_rs = np.random.RandomState(c["agent_seed"])
    out = []
    n = c["n_obs"]
    for i in range(n):
        t = MKT_OPEN + (i + 1) * c["horizon_ns"] // n
        out.append(int(oracle.observe_price("ABM", t, agent_rs, sigma_n=c["sigma_n"])))
    return out


def spec_line(c: dict) -> str:
    return (
        f"oracle {c['global_seed']} {c['agent_seed']} {float(c['r_bar'])!r} "
        f"{float(c['kappa_per_s'])!r} {float(c['sigma'])!r} {float(c['jump_intensity'])!r} "
        f"{float(c['jump_sigma'])!r} {c['horizon_ns']} {float(c['sigma_n'])!r} {c['n_obs']} "
        f"{c['jump_time_ns']} {c['jump_mag']}"
    )


def cases() -> list[dict]:
    base = dict(global_seed=1001, agent_seed=12345, r_bar=100000, kappa_per_s=0.05,
                sigma=0.0005, jump_intensity=0.0, jump_sigma=0.0,
                horizon_ns=10_000_000_000, sigma_n=1000.0, n_obs=64,
                jump_time_ns=-1, jump_mag=0)

    def v(**kw):
        c = dict(base)
        c.update(kw)
        return c

    return [
        v(name="no megashock (default lambda)"),
        v(name="different seed", global_seed=4242, agent_seed=777),
        v(name="sigma_n = 0 (no agent draw)", sigma_n=0.0),
        v(name="high fundamental vol", sigma=0.005),
        v(name="zero kappa -> rmsc04 fallback", kappa_per_s=0.0),
        # --- megashock path: 13 of 65 units live here, incl. all 6 Tier-A reactive-agent units
        v(name="megashocks fire (ji=0.5)", jump_intensity=0.5, horizon_ns=7_000_000_000),
        v(name="megashocks heavy (ji=2.0)", jump_intensity=2.0, horizon_ns=2_700_000_000),
        v(name="megashocks + jump_sigma", jump_intensity=1.5, jump_sigma=2000.0,
          horizon_ns=12_000_000_000),
        v(name="megashocks, long horizon", jump_intensity=0.5, horizon_ns=240_000_000_000,
          n_obs=128),
        # --- scheduled jump (reactive-agent intervention)
        v(name="scheduled jump mid", jump_intensity=0.5, horizon_ns=7_000_000_000,
          jump_time_ns=3_500_000_000, jump_mag=5000),
        v(name="scheduled jump early, negative", jump_intensity=0.5, horizon_ns=7_000_000_000,
          jump_time_ns=500_000_000, jump_mag=-8000),
        v(name="scheduled jump, no megashocks", jump_time_ns=5_000_000_000, jump_mag=12000),
        # --- observation past mkt_close exercises advance(mkt_close - 1)
        v(name="dense observations", n_obs=256, horizon_ns=5_000_000_000),
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--abides", default="../abides-src"))
    args = ap.parse_args(argv[1:])

    Oracle = load_real_oracle(Path(args.abides))
    cs = cases()

    proc = subprocess.run([args.binary], input="\n".join(spec_line(c) for c in cs) + "\n",
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"binary exited {proc.returncode}: {proc.stderr[:1000]}", file=sys.stderr)
        return 1
    got: dict[tuple[int, int], int] = {}
    for line in proc.stdout.split("\n"):
        if line.strip():
            ci, ii, field, value = line.split()
            if field == "obs":
                got[(int(ci), int(ii))] = int(value)

    print(f"{'case':4} {'description':34} {'checks':>9}  status")
    print("-" * 70)
    fails = 0
    first_bad: list[str] = []
    for ci, c in enumerate(cs):
        want = python_obs(Oracle, c)
        ok = sum(1 for i, w in enumerate(want) if got.get((ci, i)) == w)
        if ok != len(want):
            fails += 1
            for i, w in enumerate(want):
                g = got.get((ci, i))
                if g != w:
                    first_bad.append(f"  case {ci} obs {i}: python={w} cpp={g}")
                    break
        print(f"{ci:<4} {c['name'][:34]:34} {ok:>4}/{len(want):<4}  {'OK' if ok == len(want) else 'FAIL'}")

    total = sum(c["n_obs"] for c in cs)
    print(f"\n{len(cs) - fails}/{len(cs)} cases match the real ABIDES oracle "
          f"({total} observations compared)")
    if first_bad:
        print("\nfirst divergence per failing case:")
        print("\n".join(first_bad[:8]))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
