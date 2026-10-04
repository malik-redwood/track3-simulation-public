#!/usr/bin/env python
"""End-to-end proof: drive the all-65-unit latency check from the C++ RNG, not NumPy.

check_rng.py proves the C++ RNG matches NumPy on synthetic fixtures. This proves something
stronger and more relevant: that the C++ RNG reproduces the realized message latencies recorded
in the reference ledgers that ship with the competition -- i.e. real Tier-A data produced by the
pinned ABIDES baseline on Linux, months ago, on hardware we do not have.

For each unit it asks the C++ binary for:
  1. the latency-model sub-seed, via the verified global chain
     (1 oracle seed + 2 words of global exponential + 1 exchange seed + N agent seeds -> latency)
  2. the latency stream from that seed under the scenario's latency model
and compares against the ledger as a multiset. Send order cannot be fully recovered from a
delivery-ordered ledger at simulation shutdown, so an exact long prefix plus a permuted tail is
the expected pass -- see verify_rng_chain.py for the same treatment.

    python verify_units_cpp.py ./test_rng.exe [--repo ../../track3-simulation-public]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

# Any positive lambda works for locating the latency seed: the oracle's first megashock draw
# consumes two words regardless of its value, and only the POSITION matters here.
LAMBDA = 2.77778e-18


def run_cases(binary: str, spec: str) -> dict[tuple[int, int, str], str]:
    proc = subprocess.run([binary], input=spec, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{binary} exited {proc.returncode}: {proc.stderr[:500]}")
    out: dict[tuple[int, int, str], str] = {}
    for line in proc.stdout.split("\n"):
        if line.strip():
            ci, ii, field, value = line.split()
            out[(int(ci), int(ii), field)] = value
    return out


def ledger_latencies(path: str) -> list[int]:
    """Realized latencies in reconstructed SEND order (parent delivery seq, then message id)."""
    d = pq.read_table(path).to_pydict()
    n = len(d["seq"])
    seq_of = {(d["message_id"][i], d["dst_id"][i]): d["seq"][i] for i in range(n)}
    ev = []
    for i in range(n):
        if d["t_send_ns"][i] is None:
            continue
        par = d["causal_parent"][i]
        pseq = -1 if par is None else seq_of.get((par, d["src_id"][i]), -1)
        ev.append((pseq, d["message_id"][i], d["latency_ns"][i]))
    ev.sort()
    return [r[2] for r in ev]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--repo", default="../../track3-simulation-public")
    ap.add_argument("--repo", default(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv[1:])

    units = []
    for sp in sorted(glob.glob(os.path.join(args.repo, "units/*/scenario.json"))):
        unit = os.path.dirname(sp)
        scenario = json.load(open(sp, encoding="utf-8"))
        mt = os.path.join(unit, "message_trace.parquet")
        if not (os.path.exists(mt) and scenario.get("latency_config")):
            continue
        units.append((unit, scenario, mt))
    if not units:
        print(f"no units found under {args.repo}/units", file=sys.stderr)
        return 2

    # Pass 1: ask C++ for every latency sub-seed in one batch.
    spec = "\n".join(
        f"global_chain {s['seed']} {sum(int(c['count']) for c in s['agent_configs'])} {LAMBDA!r}"
        for _, s, _ in units
    )
    chain = run_cases(args.binary, spec + "\n")
    seeds = [int(chain[(i, 0, "latency_seed")]) for i in range(len(units))]

    # Pass 2: ask C++ for each unit's latency stream, sized to its ledger.
    observed = [ledger_latencies(mt) for _, _, mt in units]
    lines = []
    for (unit, s, _), seed, obs in zip(units, seeds, observed):
        cfg = s["latency_config"]
        p = cfg.get("params", {})
        lines.append(
            f"latency {seed} {len(obs)} {cfg['model']} "
            f"{float(p.get('mean_ns', 0.0))!r} {float(p.get('sigma', 0.0))!r} "
            f"{float(p.get('min_ns', 0.0))!r} {float(p.get('max_ns', 1e12))!r} "
            f"{float(p.get('alpha', 1.5))!r}"
        )
    streams = run_cases(args.binary, "\n".join(lines) + "\n")

    print(f"{'unit':40} {'model':12} {'result'}")
    print("-" * 92)
    fails = 0
    for ci, ((unit, s, _), obs) in enumerate(zip(units, observed)):
        pred = [int(streams[(ci, i, "value")]) for i in range(len(obs))]
        k = 0
        for a, b in zip(pred, obs):
            if a != b:
                break
            k += 1
        if k == len(obs):
            verdict = f"OK   exact {k}/{k}"
        elif sorted(pred[k:]) == sorted(obs[k:]):
            verdict = f"OK   prefix {k}/{len(obs)}, tail permuted (shutdown order)"
        else:
            verdict = f"FAIL multiset mismatch after {k}/{len(obs)}"
            fails += 1
        print(f"{os.path.basename(unit):40} {s['latency_config']['model']:12} {verdict}")

    print(f"\n{len(units) - fails}/{len(units)} units reproduced by the C++ RNG")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
