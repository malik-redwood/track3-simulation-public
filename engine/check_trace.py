#!/usr/bin/env python
"""Compare the C++ engine's trace against t3-s001's reference trace.parquet, row by row.

This is the scored output, so every one of the seven columns is compared for exact equality.
Also writes the trace to a real parquet file so it can be graded by the actual gate via
check_candidate.py -- a row-by-row match here and a gate pass there are different claims, and
both are worth having.

    python check_trace.py ./test_bootstrap [--write-parquet DIR]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REF = "../../track3-simulation-public/units/t3-s001-price-time-priority/trace.parquet"
COLS = ("t_ns", "agent_id", "msg_type", "side", "price", "size", "order_id")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--write-parquet", help="also write <DIR>/trace.parquet")
    args = ap.parse_args(argv[1:])

    proc = subprocess.run([args.binary, "--trace"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"binary exited {proc.returncode}: {proc.stderr[:1000]}", file=sys.stderr)
        return 1

    got = []
    for line in proc.stdout.split("\n"):
        if not line.strip():
            continue
        f = line.split()
        got.append((int(f[0]), int(f[1]), f[2], None if f[3] == "-" else f[3],
                    int(f[4]), int(f[5]), int(f[6])))

    t = pq.read_table(args.ref).to_pydict()
    ref = [
        (int(t["t_ns"][i]), int(t["agent_id"][i]), t["msg_type"][i], t["side"][i],
         int(t["price"][i]), int(t["size"][i]), int(t["order_id"][i]))
        for i in range(len(t["t_ns"]))
    ]

    print(f"rows: cpp={len(got)} reference={len(ref)}")
    if len(got) != len(ref):
        print("ROW COUNT MISMATCH -- Tier A requires exact equality")

    n = min(len(got), len(ref))
    bad = 0
    for i in range(n):
        if got[i] != ref[i]:
            bad += 1
            if bad <= 8:
                diffs = [f"{c}: ref={r!r} cpp={g!r}"
                         for c, r, g in zip(COLS, ref[i], got[i]) if r != g]
                print(f"row {i} MISMATCH -> " + "; ".join(diffs))

    if bad == 0 and len(got) == len(ref):
        print(f"ALL {len(got)} ROWS IDENTICAL across all 7 columns")
        # per-type tally, as a sanity check that we are not comparing something degenerate
        tally: dict[str, int] = {}
        for r in got:
            tally[r[2]] = tally.get(r[2], 0) + 1
        print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    else:
        print(f"\n{n - bad}/{n} compared rows match, {bad} mismatched")

    if args.write_parquet:
        out = Path(args.write_parquet)
        out.mkdir(parents=True, exist_ok=True)
        tbl = pa.table({
            "t_ns": pa.array([r[0] for r in got], pa.int64()),
            "agent_id": pa.array([r[1] for r in got], pa.int32()),
            "msg_type": pa.array([r[2] for r in got], pa.string()),
            "side": pa.array([r[3] for r in got], pa.string()),
            "price": pa.array([r[4] for r in got], pa.int64()),
            "size": pa.array([r[5] for r in got], pa.int64()),
            "order_id": pa.array([r[6] for r in got], pa.int64()),
        })
        pq.write_table(tbl, out / "trace.parquet", compression="snappy")
        print(f"wrote {out / 'trace.parquet'}")

    return 0 if (bad == 0 and len(got) == len(ref)) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
