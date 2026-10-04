#!/usr/bin/env python
"""Drive test_rng and compare its output against the NumPy fixtures, bit for bit.

    python gen_fixtures.py > fixtures.json
    <compile test_rng>
    python check_rng.py ./test_rng.exe

Doubles are compared as IEEE-754 bit patterns, not with a tolerance: the failure mode this
guards against is a one-word-off draw sequence, which produces plausible-looking but wrong
values that any tolerance would wave through.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path


def bits(x: float) -> str:
    """The IEEE-754 bit pattern of a double, as 16 lowercase hex digits."""
    return f"{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016x}"


def spec_lines(cases: list[dict]) -> list[str]:
    """Render the fixture cases as the flat spec test_rng reads on stdin."""
    out = []
    for c in cases:
        k = c["kind"]
        if k == "raw_words":
            out.append(f"raw_words {c['seed']} {c['n']}")
        elif k == "random_sample":
            out.append(f"random_sample {c['seed']} {c['n']}")
        elif k == "normal":
            out.append(f"normal {c['seed']} {c['n']} {c['loc']!r} {c['scale']!r}")
        elif k == "uniform":
            out.append(f"uniform {c['seed']} {c['n']} {c['low']!r} {c['high']!r}")
        elif k == "exponential":
            out.append(f"exponential {c['seed']} {c['n']} {c['scale']!r}")
        elif k == "pareto":
            out.append(f"pareto {c['seed']} {c['n']} {c['a']!r}")
        elif k == "lognormal":
            out.append(f"lognormal {c['seed']} {c['n']} {c['mean']!r} {c['sigma']!r}")
        elif k == "randint_small":
            out.append(f"randint_small {c['seed']} {c['n']} {c['low']} {c['high']}")
        elif k == "global_chain":
            out.append(f"global_chain {c['scenario_seed']} {c['n_agents']} {c['megashock_lambda']!r}")
        elif k == "noise_trader":
            out.append(
                f"noise_trader {c['seed']} {c['n']} {c['order_size_mean']!r} "
                f"{c['order_size_std']!r} {c['price_offset_ticks']}"
            )
        elif k == "latency":
            p = c["params"]
            out.append(
                f"latency {c['seed']} {c['n']} {c['model']} "
                f"{float(p.get('mean_ns', 0.0))!r} {float(p.get('sigma', 0.0))!r} "
                f"{float(p.get('min_ns', 0.0))!r} {float(p.get('max_ns', 1e12))!r} "
                f"{float(p.get('alpha', 1.5))!r}"
            )
        else:
            raise SystemExit(f"spec_lines: unhandled kind {k!r}")
    return out


def expected(cases: list[dict]) -> dict[tuple[int, int, str], str]:
    """Flatten the fixtures into {(case, item, field): value_as_text}."""
    exp: dict[tuple[int, int, str], str] = {}
    for ci, c in enumerate(cases):
        k = c["kind"]
        if k in ("raw_words", "randint_small"):
            for i, v in enumerate(c["values"]):
                exp[(ci, i, "value")] = str(int(v))
                exp[(ci, i, "pos")] = str(int(c["pos_after"][i]))
        elif k in ("random_sample", "uniform", "exponential", "pareto", "lognormal"):
            for i, v in enumerate(c["values"]):
                exp[(ci, i, "value")] = bits(v)
                exp[(ci, i, "pos")] = str(int(c["pos_after"][i]))
        elif k == "normal":
            for i, v in enumerate(c["values"]):
                exp[(ci, i, "value")] = bits(v)
                exp[(ci, i, "pos")] = str(int(c["pos_after"][i]))
                exp[(ci, i, "has_gauss")] = str(int(c["has_gauss_after"][i]))
        elif k == "global_chain":
            exp[(ci, 0, "oracle_seed")] = str(int(c["oracle_seed"]))
            exp[(ci, 0, "first_megashock_delta")] = bits(c["first_megashock_delta"])
            exp[(ci, 0, "exchange_seed")] = str(int(c["exchange_seed"]))
            for i, s in enumerate(c["agent_seeds"]):
                exp[(ci, i, "agent_seed")] = str(int(s))
            exp[(ci, 0, "latency_seed")] = str(int(c["latency_seed"]))
            exp[(ci, 0, "kernel_seed")] = str(int(c["kernel_seed"]))
        elif k == "noise_trader":
            for i, a in enumerate(c["acts"]):
                exp[(ci, i, "size")] = str(int(a["size"]))
                exp[(ci, i, "buy")] = str(int(bool(a["buy"])))
                exp[(ci, i, "offset")] = str(int(a["offset"]))
                exp[(ci, i, "pos")] = str(int(a["pos_after"]))
        elif k == "latency":
            for i, v in enumerate(c["values"]):
                exp[(ci, i, "value")] = str(int(v))
        else:
            raise SystemExit(f"expected: unhandled kind {k!r}")
    return exp


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    binary = argv[1]
    fx = json.loads(Path("fixtures.json").read_text(encoding="utf-8"))
    cases = fx["cases"]

    spec = "\n".join(spec_lines(cases)) + "\n"
    proc = subprocess.run([binary], input=spec, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"test binary exited {proc.returncode}\n{proc.stderr[:2000]}", file=sys.stderr)
        return 1

    got: dict[tuple[int, int, str], str] = {}
    for line in proc.stdout.split("\n"):
        if not line.strip():
            continue
        ci, ii, field, value = line.split()
        got[(int(ci), int(ii), field)] = value

    exp = expected(cases)
    missing = sorted(k for k in exp if k not in got)
    extra = sorted(k for k in got if k not in exp)
    wrong = sorted(k for k in exp if k in got and got[k] != exp[k])

    label = {ci: c["kind"] for ci, c in enumerate(cases)}
    per_case: dict[int, list[int]] = {ci: [0, 0] for ci in range(len(cases))}
    for k in exp:
        per_case[k[0]][1] += 1
        if k in got and got[k] == exp[k]:
            per_case[k[0]][0] += 1

    print(f"{'case':4} {'kind':16} {'checks':>9}  status")
    print("-" * 56)
    for ci in range(len(cases)):
        ok, tot = per_case[ci]
        print(f"{ci:<4} {label[ci]:16} {ok:>4}/{tot:<4}  {'OK' if ok == tot else 'FAIL'}")

    total_ok = sum(v[0] for v in per_case.values())
    total = sum(v[1] for v in per_case.values())
    print(f"\n{total_ok}/{total} checks match NumPy exactly")

    if wrong:
        print(f"\n{len(wrong)} MISMATCHES (first 12):")
        for k in wrong[:12]:
            print(f"  case {k[0]} ({label[k[0]]}) item {k[1]} {k[2]}: "
                  f"expected {exp[k]}  got {got[k]}")
    if missing:
        print(f"\n{len(missing)} fields the binary did not emit (first 8): {missing[:8]}")
    if extra:
        print(f"\n{len(extra)} unexpected fields emitted (first 8): {extra[:8]}")

    return 0 if (total_ok == total and not missing and not extra) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
