#!/usr/bin/env python
"""Compare the C++ kernel's ledger against t3-s001's reference ledger, row by row.

The order book does not exist yet, so only the startup handshake can match: rows 0..20, covering
the start-time wakeups, the market-hours request/reply handshake, and the mkt_open wakeups.
Row 21 onward is QuerySpreadMsg traffic whose replies depend on a real book.

Every field is compared exactly -- message_id, src, dst, send/recv timestamps, latency and type.
message_id is the thing to watch: it is the kernel's final tie-break, so an off-by-one there
silently reorders everything downstream.

    python check_bootstrap.py ./test_bootstrap [--rows 21]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import pyarrow.parquet as pq

REF = str(Path(__file__).resolve().parent.parent / "units/t3-s001-price-time-priority//message_trace.parquet")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--rows", type=int, default=21)
    ap.add_argument("--ref", default=REF)
    args = ap.parse_args(argv[1:])

    d = pq.read_table(args.ref).to_pydict()
    ref = []
    for i in range(len(d["seq"])):
        ref.append((
            int(d["seq"][i]), int(d["message_id"][i]), int(d["src_id"][i]), int(d["dst_id"][i]),
            None if d["t_send_ns"][i] is None else int(d["t_send_ns"][i]),
            int(d["t_recv_ns"][i]), int(d["latency_ns"][i]), d["msg_type"][i],
        ))
    ref.sort(key=lambda r: r[0])

    proc = subprocess.run([args.binary], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"binary exited {proc.returncode}: {proc.stderr[:1000]}", file=sys.stderr)
        return 1
    # Binary emits all 10 message_trace columns:
    #   seq t_recv_ns t_send_ns latency_ns src dst message_id msg_type order_id causal_parent
    got = []
    for line in proc.stdout.split("\n"):
        if not line.strip():
            continue
        f = line.split()
        got.append((
            int(f[0]), int(f[6]), int(f[4]), int(f[5]),
            None if f[2] == "-" else int(f[2]),
            int(f[1]), int(f[3]), f[7],
        ))

    n = min(args.rows, len(ref), len(got))
    cols = ("seq", "message_id", "src", "dst", "t_send", "t_recv", "latency", "msg_type")
    bad = 0
    print(f"comparing {n} rows (C++ produced {len(got)}, reference has {len(ref)})\n")
    for i in range(n):
        if ref[i] != got[i]:
            bad += 1
            if bad <= 5:
                diffs = [f"{c}: ref={r} cpp={g}"
                         for c, r, g in zip(cols, ref[i], got[i]) if r != g]
                print(f"row {i} MISMATCH -> " + "; ".join(diffs))
    if bad == 0:
        print(f"all {n} rows identical: message_id, src, dst, t_send, t_recv, latency, msg_type")
        tail = got[n - 1]
        print(f"last matched row: seq={tail[0]} id={tail[1]} {tail[2]}->{tail[3]} {tail[7]}")
    else:
        print(f"\n{n - bad}/{n} rows match, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
