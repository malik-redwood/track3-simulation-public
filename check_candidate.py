#!/usr/bin/env python
"""Local Track-3 correctness harness — no Docker required.

Mirrors the official g3 domain-semantics gate (`qfbench2_track_simulation.scoring._g3_body`)
against the reference traces that ship inside `units/`, so you can develop a trace-exact engine
without building or running a container.

It reads the SAME policy the official gate reads — `card.toml`, not `scenario.json` — via a
faithful copy of `scoring._CardPolicy`, and calls the SAME `semantics` / `stylized_facts` /
`batch` functions. It is not a reimplementation of the scoring math.

Usage
-----
    # sanity-check the harness itself: the shipped references, graded as the candidate
    python check_candidate.py --self-test

    # grade your engine's output
    python check_candidate.py --candidate run_outputs/

    # one unit, with full breach detail
    python check_candidate.py --candidate run_outputs/ --unit t3-s001-price-time-priority -v

Candidate layout (same as the real harness writes):
    <candidate>/<slug>/trace.parquet
    <candidate>/<slug>/message_trace.parquet        (when the card requires a ledger)
    <candidate>/<slug>/events.json                  (optional; n_events is cross-checked if present)
    <candidate>/<slug>/<sub>/{trace,message_trace}.parquet   (batch units)
    <candidate>/<slug>/batch_events.json                     (batch units)

What this does NOT cover: g0 integrity, host-measured timing, the output-sanitize path (POSIX
only), and the sealed scenarios. Passing here is necessary, not sufficient.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from qfbench2_common.scoring import stylized_facts
from qfbench2_track_simulation import batch as _batch
from qfbench2_track_simulation import semantics
from qfbench2_track_simulation.limits import requires_message_ledger

# Copied from scoring.py so a drift between the two is visible rather than silent.
_FAMILY_NUM: dict[str, int] = {
    "matching-engine-semantics": 1,
    "agent-mix": 2,
    "latency-profile": 3,
    "oracle-noise": 4,
    "calibration-stylized-facts": 5,
    "throughput-scale": 6,
    "exchange-protocol": 7,
    "reactive-agent": 8,
}
_DEFAULT_CEILINGS: dict[str, float] = {
    "ks": 0.08,
    "acf_abs_l2": 0.12,
    "hill_abs": 1.5,
    "depth_js": 0.10,
}


@dataclass
class Policy:
    """Faithful copy of `scoring._CardPolicy` — the card is authoritative, not scenario.json."""

    family: int
    tier: str | None
    ceilings: dict[str, float]
    timestamp_tolerance_ns: int
    kendall_tau_floor: float
    spread_bps_tolerance: float
    requires_message_ledger: bool
    is_batch: bool

    @classmethod
    def load(cls, unit_dir: Path) -> "Policy":
        cp = unit_dir / "card.toml"
        card = tomllib.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
        params = card.get("scoring", {}).get("params", {})
        ceilings = dict(_DEFAULT_CEILINGS)
        ceilings.update(params.get("stylized_fact_ceilings", {}))
        return cls(
            family=_FAMILY_NUM.get(card.get("task", {}).get("scenario_family", ""), 0),
            tier=params.get("semantic_tier"),
            ceilings=ceilings,
            timestamp_tolerance_ns=int(
                params.get(
                    "timestamp_tolerance_ns", semantics.DEFAULT_TIMESTAMP_TOLERANCE_NS
                )
            ),
            kendall_tau_floor=float(
                params.get("kendall_tau_floor", semantics.DEFAULT_KENDALL_TAU_FLOOR)
            ),
            spread_bps_tolerance=float(params.get("spread_bps_tolerance", 10.0)),
            requires_message_ledger=requires_message_ledger(card),
            is_batch=_batch.batch_meta(unit_dir) is not None,
        )


@dataclass
class Result:
    slug: str
    policy: Policy
    ok: bool = False
    stage: str = ""
    breaches: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def tier_label(self) -> str:
        t = self.policy.tier or ("A" if self.policy.family in semantics.TIER_A_FAMILIES else "B")
        return str(t).upper()


def _fail(res: Result, stage: str, breaches) -> Result:
    res.ok, res.stage = False, stage
    res.breaches = [str(b) for b in (breaches if isinstance(breaches, list) else [breaches])]
    return res


def grade_unit(unit_dir: Path, cand_dir: Path) -> Result:
    """Grade one unit, classifying exceptions exactly as `scoring._g3_domain_semantics` does.

    The classification matters: a `ValueError` out of `mid_price_series` (which is what duplicate
    `(t_ns, side)` QUOTE_UPDATE rows produce) is caught by the official gate and turned into
    T3_PARSE_ERROR — an inadmissible participant failure reported as "unreadable or malformed
    candidate output", which points at file corruption rather than the real cause.
    """
    try:
        return _grade_unit_body(unit_dir, cand_dir)
    except semantics.ReferenceIncomplete as exc:
        return _fail(Result(unit_dir.name, Policy.load(unit_dir)), "ORGANIZER_reference_incomplete", str(exc))
    except semantics.NonfiniteStatistic as exc:
        return _fail(Result(unit_dir.name, Policy.load(unit_dir)), "ORGANIZER_nonfinite_statistic", str(exc))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        return _fail(
            Result(unit_dir.name, Policy.load(unit_dir)),
            "parse_error",
            f"{type(exc).__name__}: {exc}  "
            "(the official gate reports this as T3_PARSE_ERROR / 'unreadable or malformed "
            "candidate output' -- check for duplicate (t_ns, side) QUOTE_UPDATE rows)",
        )


def _grade_unit_body(unit_dir: Path, cand_dir: Path) -> Result:
    """Mirrors `_g3_body`, in the same order, with the same arguments."""
    policy = Policy.load(unit_dir)
    res = Result(slug=unit_dir.name, policy=policy)

    # ---- batch units: g3 is the isolation gate only, plus the aggregate cross-check ----
    if policy.is_batch:
        if not cand_dir.exists():
            return _fail(res, "missing", f"no candidate directory {cand_dir}")
        ok, failures = _batch.score_isolation(
            unit_dir,
            cand_dir,
            policy.family,
            policy.tier,
            policy.ceilings,
            timestamp_tolerance_ns=policy.timestamp_tolerance_ns,
            kendall_tau_floor=policy.kendall_tau_floor,
        )
        if not ok:
            return _fail(res, "batch_isolation", [json.dumps(f) for f in failures[:3]])
        agg_ok, info = _batch.check_aggregate(cand_dir, _batch.load_subs(unit_dir))
        if not agg_ok:
            return _fail(res, "batch_aggregate", [json.dumps(info)])
        res.ok, res.stage, res.extra = True, "ok", info
        return res

    # ---- single-scenario units ----
    ref_path, cand_path = unit_dir / "trace.parquet", cand_dir / "trace.parquet"
    if not ref_path.exists():
        return _fail(res, "organizer", f"reference trace missing: {ref_path}")
    if not cand_path.exists():
        return _fail(res, "missing", f"missing candidate trace.parquet in {cand_dir}")
    ref, cand = pd.read_parquet(ref_path), pd.read_parquet(cand_path)
    res.extra["rows"] = f"{len(cand)}/{len(ref)}"

    # anti-inflation cross-check on the sidecar, when one is present
    ev = cand_dir / "events.json"
    if ev.exists():
        reported = json.loads(ev.read_text(encoding="utf-8")).get("n_events")
        if reported is not None and int(float(reported)) != len(cand):
            return _fail(
                res,
                "events_json",
                f"events.json n_events={int(float(reported))} != trace rows {len(cand)}",
            )

    # (a) semantic regression, with the CARD's tolerances
    ok, breaches = semantics.semantic_regression_pass(
        cand,
        ref,
        family=policy.family,
        tier=policy.tier,
        ceilings=policy.ceilings,
        spread_bps_tolerance=policy.spread_bps_tolerance,
        timestamp_tolerance_ns=policy.timestamp_tolerance_ns,
        kendall_tau_floor=policy.kendall_tau_floor,
    )
    if not ok:
        return _fail(res, f"semantic_tier_{res.tier_label}", breaches)

    # (b) stylized-fact admissibility — Family 5 only. Depth hists are KEYWORDS: passed
    # positionally they land on cand/ref_intraday_vol and depth_js is silently never computed.
    if policy.family == 5:
        report = stylized_facts.stylized_fact_report(
            semantics.mid_price_series(cand).to_numpy(dtype=float),
            semantics.mid_price_series(ref).to_numpy(dtype=float),
            cand_depth_hist=semantics.depth_histogram(cand),
            ref_depth_hist=semantics.depth_histogram(ref),
        )
        res.extra["stylized_facts"] = {k: round(v, 6) for k, v in report.items()}
        adm, sf = stylized_facts.admissible(report, policy.ceilings)
        if not adm:
            return _fail(res, "stylized_facts", sf)

    # (c) message ledger, mandatory by card declaration
    if not policy.requires_message_ledger:
        res.ok, res.stage = True, "ok (no ledger required)"
        return res

    ref_msg_path = unit_dir / "message_trace.parquet"
    if not ref_msg_path.exists():
        return _fail(res, "organizer", "card requires a ledger but no reference ships one")
    cand_msg_path = cand_dir / "message_trace.parquet"
    if not cand_msg_path.exists():
        return _fail(res, "ledger_missing", "card requires message_trace.parquet; none emitted")
    ref_msg, cand_msg = pd.read_parquet(ref_msg_path), pd.read_parquet(cand_msg_path)
    res.extra["ledger_rows"] = f"{len(cand_msg)}/{len(ref_msg)}"

    ok_m, br_m = semantics.check_message_semantics(cand_msg)
    if not ok_m:
        return _fail(res, "message_semantics", br_m)
    ok_r, br_r = semantics.check_message_reference(cand_msg, ref_msg)
    if not ok_r:
        return _fail(res, "message_reference", br_r)
    if policy.family == 7:
        ok_p, br_p = semantics.check_protocol_fidelity(cand_msg, ref_msg)
        if not ok_p:
            return _fail(res, "protocol_fidelity", br_p)

    res.ok, res.stage = True, "ok"
    return res


def build_self_test_tree(units: list[Path], root: Path) -> Path:
    """Materialize the shipped references as a candidate tree, so the harness can grade itself.

    Single units are graded in place (the reference dir already has the right shape). Batch units
    need a `<slug>/<sub>/` layout plus a synthesized `batch_events.json`, assembled here from
    `checks/reference_data/`.
    """
    for unit in units:
        if _batch.batch_meta(unit) is None:
            continue
        out = root / unit.name
        out.mkdir(parents=True, exist_ok=True)
        per_scenario, total = [], 0
        for entry in _batch.load_subs(unit):
            sub = entry["sub"]
            src, dst = unit / "checks" / "reference_data" / sub, out / sub
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("trace.parquet", "message_trace.parquet"):
                if (src / name).exists():
                    shutil.copyfile(src / name, dst / name)
            rows = len(pd.read_parquet(dst / "trace.parquet"))
            per_scenario.append({"sub": sub, "n_events": rows})
            total += rows
        wall = 1.0
        (out / "batch_events.json").write_text(
            json.dumps(
                {
                    "n_scenarios": len(per_scenario),
                    "total_events": total,
                    "wall_clock_sec": wall,
                    "events_per_sec": total / wall,
                    "per_scenario": per_scenario,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="track3-simulation-public clone root (default: cwd)")
    ap.add_argument("--candidate", help="directory holding <slug>/trace.parquet per unit")
    ap.add_argument("--self-test", action="store_true", help="grade the shipped references as the candidate")
    ap.add_argument("--unit", action="append", default=[], help="only this unit (repeatable)")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every breach, not just the first")
    args = ap.parse_args(argv)

    if not args.candidate and not args.self_test:
        ap.error("pass --candidate DIR or --self-test")

    repo = Path(args.repo).resolve()
    units_root = repo / "units"
    if not units_root.is_dir():
        print(f"error: {units_root} not found; use --repo to point at the clone root", file=sys.stderr)
        return 2

    units = sorted(p for p in units_root.iterdir() if (p / "card.toml").exists())
    if args.unit:
        wanted = set(args.unit)
        units = [u for u in units if u.name in wanted]
        missing = wanted - {u.name for u in units}
        if missing:
            print(f"error: unknown unit(s): {sorted(missing)}", file=sys.stderr)
            return 2
    if not units:
        print("error: no units found", file=sys.stderr)
        return 2

    tmp: tempfile.TemporaryDirectory | None = None
    if args.self_test:
        tmp = tempfile.TemporaryDirectory(prefix="t3-selftest-")
        synth = build_self_test_tree(units, Path(tmp.name))

        def cand_for(unit: Path) -> Path:
            return synth / unit.name if _batch.batch_meta(unit) else unit
    else:
        cand_root = Path(args.candidate).resolve()

        def cand_for(unit: Path) -> Path:
            return cand_root / unit.name

    results: list[Result] = []
    try:
        for unit in units:
            try:
                res = grade_unit(unit, cand_for(unit))
            except semantics.OrganizerSemanticFault as exc:
                res = _fail(Result(unit.name, Policy.load(unit)), "ORGANIZER_FAULT", str(exc))
            except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
                res = _fail(Result(unit.name, Policy.load(unit)), f"ERROR {type(exc).__name__}", str(exc))
            results.append(res)
            mark = "PASS" if res.ok else "FAIL"
            fam = f"F{res.policy.family}/{res.tier_label}"
            kind = "batch" if res.policy.is_batch else ("ledger" if res.policy.requires_message_ledger else "no-ldg")
            print(f"[{mark}] {res.slug:40} {fam:6} {kind:7} {res.stage}")
            if not res.ok:
                for b in (res.breaches if args.verbose else res.breaches[:1]):
                    print(f"         - {b[:300]}")
    finally:
        if tmp is not None:
            tmp.cleanup()

    n_pass = sum(1 for r in results if r.ok)
    print(f"\n{n_pass}/{len(results)} units pass")
    by_stage: dict[str, int] = {}
    for r in results:
        if not r.ok:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
    for stage, n in sorted(by_stage.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d} failing at {stage}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {
                        "unit": r.slug,
                        "ok": r.ok,
                        "family": r.policy.family,
                        "tier": r.tier_label,
                        "is_batch": r.policy.is_batch,
                        "requires_message_ledger": r.policy.requires_message_ledger,
                        "stage": r.stage,
                        "breaches": r.breaches,
                        **r.extra,
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report -> {args.json}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
