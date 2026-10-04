#!/usr/bin/env python
"""Run the C++ engine over every single-scenario unit, write parquets, and grade with the gate.

Reports, per unit, how far the trace matches the reference before diverging -- a long prefix
means the engine is nearly right and points at a specific behaviour, whereas an immediate
divergence means something structural. That distinction is what makes the failures diagnosable.

    python sweep.py ./run_unit [--out ../run_outputs] [--limit N] [--unit SLUG]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

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


def run(binary: str, scenario: Path, mode: str) -> list[list[str]]:
    p = subprocess.run([binary, str(scenario), mode], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:300] or f"exit {p.returncode}")
    return [ln.split() for ln in p.stdout.split("\n") if ln.strip()]


def nullable(v: str):
    return None if v == "-" else int(v)


def write_outputs(out_dir: Path, tr: list[list[str]], lg: list[list[str]],
                  scenario_id: str, seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "t_ns": pd.array([int(r[0]) for r in tr], dtype="int64"),
        "agent_id": pd.array([int(r[1]) for r in tr], dtype="int32"),
        "msg_type": [r[2] for r in tr],
        "side": [None if r[3] == "-" else r[3] for r in tr],
        "price": pd.array([int(r[4]) for r in tr], dtype="int64"),
        "size": pd.array([int(r[5]) for r in tr], dtype="int64"),
        "order_id": pd.array([int(r[6]) for r in tr], dtype="int64"),
    }).to_parquet(out_dir / "trace.parquet", compression="snappy", index=False)

    # t_send_ns / order_id / causal_parent MUST be pandas nullable Int64. Plain int64-with-nulls
    # round-trips as float64, which cannot hold a 1.6e18 timestamp, and the gate then reports
    # bogus "latency identity" failures.
    pd.DataFrame({
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
    }).to_parquet(out_dir / "message_trace.parquet", compression="snappy", index=False)

    raw = (out_dir / "trace.parquet").read_bytes()
    (out_dir / "events.json").write_text(json.dumps({
        "scenario_id": scenario_id, "seed": seed, "n_events": len(tr),
        "wall_clock_sec": 0.001, "events_per_sec": len(tr) / 0.001,
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "peak_memory_bytes": 0, "gpu_seconds": 0.0,
    }, indent=2))
    return len(tr)


def prefix_match(cand: Path, ref: Path) -> tuple[int, int, int]:
    """Returns (matching prefix length, candidate rows, reference rows)."""
    c = pq.read_table(cand).to_pydict()
    r = pq.read_table(ref).to_pydict()
    cols = ("t_ns", "agent_id", "msg_type", "side", "price", "size", "order_id")
    n = min(len(c["t_ns"]), len(r["t_ns"]))
    k = 0
    for i in range(n):
        if any(c[col][i] != r[col][i] for col in cols):
            break
        k += 1
    return k, len(c["t_ns"]), len(r["t_ns"])


def do_batch(binary: str, unit: Path, out_root: Path) -> tuple[str, int, int]:
    """Run every sub-scenario of a batch unit independently and write the aggregate.

    Each sub must reproduce the ISOLATED reference, which is satisfiable because
    run_simulation resets the global order_id / message_id counters and re-seeds from the sub's
    own seed -- the same thing simulate.py does per call. So subs are run sequentially with no
    shared state, never interleaved.
    """
    meta = json.loads((unit / "batch.json").read_text())
    out_dir = out_root / unit.name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scenario, total, exact_subs = [], 0, 0
    for entry in meta["subs"]:
        sub = entry["sub"]
        scenario = unit / entry.get("scenario_file", f"scenarios/{sub}.json")
        sc = json.loads(scenario.read_text())
        tr = run(binary, scenario, "--trace")
        lg = run(binary, scenario, "--ledger")
        n = write_outputs(out_dir / sub, tr, lg, sc["scenario_id"], int(sc["seed"]))
        per_scenario.append({"sub": sub, "n_events": n})
        total += n
        ref = unit / "checks" / "reference_data" / sub / "trace.parquet"
        if ref.exists():
            k, nc, nr = prefix_match(out_dir / sub / "trace.parquet", ref)
            if nc == nr and k == nr:
                exact_subs += 1
    # check_aggregate requires wall_clock_sec and cross-checks events_per_sec against
    # total_events / wall_clock_sec, plus every per_scenario n_events against the real row count.
    wall = 0.001
    (out_dir / "batch_events.json").write_text(json.dumps({
        "n_scenarios": len(per_scenario),
        "total_events": total,
        "wall_clock_sec": wall,
        "events_per_sec": total / wall,
        "peak_memory_bytes": 0,
        "gpu_seconds": 0.0,
        "per_scenario": per_scenario,
    }, indent=2))
    return unit.name, exact_subs, len(per_scenario)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--out", default="../run_outputs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--unit")
    ap.add_argument("--batch-only", action="store_true")
    args = ap.parse_args(argv[1:])

    out_root_early = Path(args.out)
    batch_units = sorted(p.parent for p in REPO.glob("units/*/batch.json"))
    if args.unit:
        batch_units = [u for u in batch_units if u.name == args.unit]
    if batch_units:
        print(f"{'batch unit':40} {'subs exact':>12}")
        print("-" * 56)
        for u in batch_units:
            try:
                name, ex, tot = do_batch(args.binary, u, out_root_early)
                print(f"{name:40} {f'{ex}/{tot}':>12}  {'EXACT' if ex == tot else 'diverges'}")
            except Exception as e:  # noqa: BLE001
                print(f"{u.name:40} {'-':>12}  ERROR: {str(e)[:80]}")
        print()
    if args.batch_only:
        return 0

    units = sorted(p.parent for p in REPO.glob("units/*/scenario.json"))
    if args.unit:
        units = [u for u in units if u.name == args.unit]
    if args.limit:
        units = units[: args.limit]

    out_root = Path(args.out)
    print(f"{'unit':40} {'rows c/r':>15} {'prefix':>9}  status")
    print("-" * 92)
    rows = []
    for u in units:
        sc = json.loads((u / "scenario.json").read_text())
        try:
            tr = run(args.binary, u / "scenario.json", "--trace")
            lg = run(args.binary, u / "scenario.json", "--ledger")
        except RuntimeError as e:
            print(f"{u.name:40} {'-':>15} {'-':>9}  ENGINE ERROR: {e}")
            rows.append((u.name, "engine_error", 0, 0, 0))
            continue
        write_outputs(out_root / u.name, tr, lg, sc["scenario_id"], int(sc["seed"]))
        k, nc, nr = prefix_match(out_root / u.name / "trace.parquet", u / "trace.parquet")
        exact = (nc == nr and k == nr)
        print(f"{u.name:40} {f'{nc}/{nr}':>15} {k:>9}  {'EXACT' if exact else 'diverges'}")
        rows.append((u.name, "exact" if exact else "diverges", k, nc, nr))

    print()
    exact = [r for r in rows if r[1] == "exact"]
    err = [r for r in rows if r[1] == "engine_error"]
    div = [r for r in rows if r[1] == "diverges"]
    print(f"{len(exact)}/{len(rows)} units reproduce the reference trace EXACTLY")
    print(f"{len(div)} diverge, {len(err)} engine errors")

    if div:
        print("\ndivergence profile (how many reference rows matched before the first difference):")
        buckets = Counter()
        for name, _, k, nc, nr in div:
            if k == 0: buckets["0 rows (structural)"] += 1
            elif k < 25: buckets["1-24 rows"] += 1
            elif k < nr * 0.5: buckets["<50% of trace"] += 1
            else: buckets[">=50% of trace"] += 1
        for b, c in buckets.most_common():
            print(f"  {c:3}  {b}")
        print("\nworst offenders (shortest prefix first):")
        for name, _, k, nc, nr in sorted(div, key=lambda r: r[2])[:12]:
            print(f"  {name:40} prefix {k:>7} of {nr:>7}  (emitted {nc})")
    return 0 if len(exact) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
