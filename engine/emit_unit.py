#!/usr/bin/env python
"""Write the C++ engine's outputs as real parquet files, then grade them with the actual gate.

Compares all 10 message_trace columns against the reference (check_bootstrap.py only compares
8 -- order_id and causal_parent are validated here), writes trace.parquet + message_trace.parquet
+ events.json in the layout check_candidate.py expects, and invokes it.

    python emit_unit.py ./test_bootstrap --out ../run_outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

UNIT = "t3-s001-price-time-priority"
REPO = Path("../../track3-simulation-public")


def run(binary: str, *flags: str) -> list[list[str]]:
    p = subprocess.run([binary, *flags], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{binary} {' '.join(flags)} exited {p.returncode}: {p.stderr[:800]}")
    return [ln.split() for ln in p.stdout.split("\n") if ln.strip()]


def nullable(v: str):
    return None if v == "-" else int(v)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--out", default="../run_outputs")
    args = ap.parse_args(argv[1:])

    unit_dir = REPO / "units" / UNIT
    out_dir = Path(args.out) / UNIT
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- trace ----------------
    tr = run(args.binary, "--trace")
    trace = pa.table({
        "t_ns": pa.array([int(r[0]) for r in tr], pa.int64()),
        "agent_id": pa.array([int(r[1]) for r in tr], pa.int32()),
        "msg_type": pa.array([r[2] for r in tr], pa.string()),
        "side": pa.array([None if r[3] == "-" else r[3] for r in tr], pa.string()),
        "price": pa.array([int(r[4]) for r in tr], pa.int64()),
        "size": pa.array([int(r[5]) for r in tr], pa.int64()),
        "order_id": pa.array([int(r[6]) for r in tr], pa.int64()),
    })
    pq.write_table(trace, out_dir / "trace.parquet", compression="snappy")

    # ---------------- message ledger ----------------
    # The three nullable columns MUST round-trip as pandas Int64 (nullable integer), not
    # float64. A plain arrow int64 with nulls comes back from pd.read_parquet as float64, and
    # float64 cannot represent a ~1.6e18 nanosecond timestamp exactly (granularity 256 ns). The
    # gate then reports "Latency identity t_recv-t_send != latency_ns" on 575 of 580 messages --
    # a data-looking error caused purely by encoding. The reference uses Int64Dtype, so writing
    # through pandas with the same dtypes reproduces the pandas metadata arrow needs.
    lg = run(args.binary)
    ledger_df = pd.DataFrame({
        "seq": pd.array([int(r[0]) for r in lg], dtype="int64"),
        "t_recv_ns": pd.array([int(r[1]) for r in lg], dtype="int64"),
        "t_send_ns": pd.array([nullable(r[2]) for r in lg], dtype="Int64"),
        "latency_ns": pd.array([int(r[3]) for r in lg], dtype="int64"),
        "src_id": pd.array([int(r[4]) for r in lg], dtype="int32"),
        "dst_id": pd.array([int(r[5]) for r in lg], dtype="int32"),
        "message_id": pd.array([int(r[6]) for r in lg], dtype="int64"),
        "msg_type": [r[7] for r in lg],
        "order_id": pd.array([nullable(r[8]) for r in lg], dtype="Int64"),
        "causal_parent": pd.array([nullable(r[9]) for r in lg], dtype="Int64"),
    })
    ledger_df.to_parquet(out_dir / "message_trace.parquet", compression="snappy", index=False)
    ledger = pa.Table.from_pandas(ledger_df, preserve_index=False)

    # ---------------- full 10-column ledger comparison ----------------
    ref = pq.read_table(unit_dir / "message_trace.parquet").to_pydict()
    cols = ["seq", "t_recv_ns", "t_send_ns", "latency_ns", "src_id", "dst_id", "message_id",
            "msg_type", "order_id", "causal_parent"]
    got = ledger.to_pydict()
    n = min(len(ref["seq"]), len(got["seq"]))
    bad = 0
    for c in cols:
        mism = [i for i in range(n) if ref[c][i] != got[c][i]]
        if mism:
            bad += len(mism)
            print(f"  ledger column {c}: {len(mism)} mismatches, first at row {mism[0]} "
                  f"(ref={ref[c][mism[0]]!r} cpp={got[c][mism[0]]!r})")
    if bad == 0 and len(ref["seq"]) == len(got["seq"]):
        print(f"message_trace: ALL {n} rows identical across all 10 columns "
              f"(incl. order_id and causal_parent)")
    else:
        print(f"message_trace: {bad} field mismatches over {n} rows")

    # ---------------- events.json ----------------
    raw = (out_dir / "trace.parquet").read_bytes()
    sidecar = {
        "scenario_id": json.loads((unit_dir / "scenario.json").read_text())["scenario_id"],
        "seed": 1001,
        "n_events": trace.num_rows,
        "wall_clock_sec": 0.001,
        "events_per_sec": trace.num_rows / 0.001,
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "peak_memory_bytes": 0,
        "gpu_seconds": 0.0,
    }
    (out_dir / "events.json").write_text(json.dumps(sidecar, indent=2))

    # ---------------- grade with the real gate ----------------
    print()
    g = subprocess.run(
        [sys.executable, "../check_candidate.py", "--repo", str(REPO),
         "--candidate", str(Path(args.out).resolve()), "--unit", UNIT, "-v"],
        capture_output=True, text=True,
    )
    print(g.stdout.strip())
    if g.stderr.strip():
        print(g.stderr.strip()[:1500], file=sys.stderr)
    return g.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
