#!/usr/bin/env python
"""Generate ground-truth RNG fixtures from NumPy's legacy RandomState.

The C++ engine must reproduce these streams bit for bit. NumPy guarantees RandomState stream
stability across versions, so these fixtures are a durable contract rather than a snapshot of
one install.

Each case records the exact call sequence and the values produced, plus the MT19937 word
position after each call -- the position is what catches a wrong word-consumption count even
when the returned value happens to look plausible.

    python gen_fixtures.py > fixtures.json
"""

from __future__ import annotations

import json
import sys

import numpy as np


def pos(rs: np.random.RandomState) -> int:
    """MT19937 words consumed so far (624 = buffer exhausted, pre-regeneration)."""
    return int(rs.get_state()[2])


def gauss_state(rs: np.random.RandomState) -> tuple[int, float]:
    st = rs.get_state()
    return int(st[3]), float(st[4])


def case_raw_words(seed: int, n: int) -> dict:
    """randint(0, 2**32, dtype=uint64) -- must consume exactly ONE word and return it."""
    rs = np.random.RandomState(seed)
    vals, positions = [], []
    for _ in range(n):
        vals.append(int(rs.randint(low=0, high=2**32, dtype="uint64")))
        positions.append(pos(rs))
    return {"kind": "raw_words", "seed": seed, "n": n, "values": vals, "pos_after": positions}


def case_normal(seed: int, n: int, loc: float, scale: float) -> dict:
    """Marsaglia polar with a cached second value: variable word cost, parity matters."""
    rs = np.random.RandomState(seed)
    vals, positions, cached = [], [], []
    for _ in range(n):
        vals.append(float(rs.normal(loc, scale)))
        positions.append(pos(rs))
        cached.append(gauss_state(rs)[0])
    return {
        "kind": "normal", "seed": seed, "n": n, "loc": loc, "scale": scale,
        "values": vals, "pos_after": positions, "has_gauss_after": cached,
    }


def case_simple(kind: str, seed: int, n: int, **kw) -> dict:
    rs = np.random.RandomState(seed)
    fn = {
        "uniform": lambda: rs.uniform(kw["low"], kw["high"]),
        "exponential": lambda: rs.exponential(kw["scale"]),
        "pareto": lambda: rs.pareto(kw["a"]),
        "lognormal": lambda: rs.lognormal(kw["mean"], kw["sigma"]),
        "randint_small": lambda: int(rs.randint(kw["low"], kw["high"])),
        "random_sample": lambda: float(rs.random_sample()),
    }[kind]
    vals, positions = [], []
    for _ in range(n):
        v = fn()
        vals.append(float(v) if not isinstance(v, int) else v)
        positions.append(pos(rs))
    return {"kind": kind, "seed": seed, "n": n, **kw,
            "values": vals, "pos_after": positions}


def case_global_chain(scenario_seed: int, n_agents: int) -> dict:
    """The verified config.py chain: N+2 randint sub-seeds around one global exponential.

    word 1        -> oracle symbol RandomState seed
    words 2,3     -> np.random.exponential(1/megashock_lambda_a)   (oracle __init__, GLOBAL)
    word 4        -> ExchangeAgent seed
    words 5..4+N  -> agent seeds, construction order
    word 5+N      -> latency model seed
    word 6+N      -> kernel seed
    """
    np.random.seed(scenario_seed)
    oracle_seed = int(np.random.randint(0, 2**32, dtype="uint64"))
    lam = 2.77778e-18  # the rmsc04 fallback used when jump_intensity is 0
    first_megashock = float(np.random.exponential(scale=1.0 / lam))
    exchange_seed = int(np.random.randint(0, 2**32, dtype="uint64"))
    agent_seeds = [int(np.random.randint(0, 2**32, dtype="uint64")) for _ in range(n_agents)]
    latency_seed = int(np.random.randint(0, 2**32, dtype="uint64"))
    kernel_seed = int(np.random.randint(0, 2**32, dtype="uint64"))
    return {
        "kind": "global_chain", "scenario_seed": scenario_seed, "n_agents": n_agents,
        "megashock_lambda": lam,
        "oracle_seed": oracle_seed, "first_megashock_delta": first_megashock,
        "exchange_seed": exchange_seed, "agent_seeds": agent_seeds,
        "latency_seed": latency_seed, "kernel_seed": kernel_seed,
    }


def case_noise_trader(seed: int, n: int, mean: float, std: float, offset_ticks: int) -> dict:
    """NoiseTrader.act(): normal -> randint(0,2) -> randint(0,offset+1), three draws per act."""
    rs = np.random.RandomState(seed)
    acts = []
    for _ in range(n):
        size = int(max(1, round(rs.normal(mean, std))))
        buy = bool(rs.randint(0, 2))
        off = int(rs.randint(0, offset_ticks + 1))
        acts.append({"size": size, "buy": buy, "offset": off, "pos_after": pos(rs)})
    return {"kind": "noise_trader", "seed": seed, "n": n, "order_size_mean": mean,
            "order_size_std": std, "price_offset_ticks": offset_ticks, "acts": acts}


def case_latency(seed: int, n: int, model: str, params: dict) -> dict:
    """config.py ScenarioLatencyModel.get_latency, including banker's rounding."""
    p = params
    mean = float(p.get("mean_ns", 0.0)); sigma = float(p.get("sigma", 0.0))
    lo = float(p.get("min_ns", 0.0)); hi = float(p.get("max_ns", 1e12))
    alpha = float(p.get("alpha", 1.5))
    mu = float(np.log(mean)) if mean > 0 else 0.0
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        if model == "log_normal":
            v = rs.lognormal(mean=mu, sigma=sigma)
        elif model == "uniform":
            v = rs.uniform(lo, hi)
        elif model == "pareto":
            v = (lo if lo > 0 else 1.0) * (1.0 + rs.pareto(alpha))
        else:
            v = mean
        vals.append(int(round(float(np.clip(v, lo, hi)))))
    return {"kind": "latency", "seed": seed, "n": n, "model": model, "params": p, "values": vals}


def main() -> int:
    fixtures = {
        "numpy_version": np.__version__,
        "cases": [
            case_raw_words(1001, 16),
            case_raw_words(0, 8),
            case_raw_words(4294967295, 8),
            case_simple("random_sample", 7, 8),
            case_normal(7, 12, 0.0, 1.0),
            case_normal(1001, 12, 10.0, 2.0),
            case_normal(12345, 40, 100.0, 15.0),      # long run: exercises the rejection loop
            case_simple("uniform", 7, 8, low=100.0, high=2000.0),
            case_simple("exponential", 7, 8, scale=3.0),
            case_simple("pareto", 7, 8, a=1.5),
            case_simple("lognormal", 7, 8, mean=6.2, sigma=0.5),
            case_simple("randint_small", 7, 64, low=0, high=2),
            case_simple("randint_small", 7, 64, low=0, high=6),   # rng=5, mask=7 -> rejections
            case_simple("randint_small", 99, 64, low=0, high=10),
            case_global_chain(1001, 4),
            case_global_chain(42, 20),
            case_noise_trader(842071566, 20, 10.0, 2.0, 5),       # agent 1 of t3-s001
            case_latency(99271950, 32, "uniform", {"min_ns": 100.0, "max_ns": 2000.0, "mean_ns": 500.0}),
            case_latency(12345, 32, "log_normal", {"mean_ns": 500.0, "sigma": 0.5, "min_ns": 100.0, "max_ns": 1e6}),
            case_latency(777, 32, "pareto", {"min_ns": 200.0, "max_ns": 1e7, "alpha": 1.5}),
        ],
    }
    json.dump(fixtures, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
