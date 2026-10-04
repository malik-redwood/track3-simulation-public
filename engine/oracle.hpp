// oracle.hpp — SparseMeanRevertingOracle, bit-exact against the patched ABIDES original.
//
// Mirrors abides-markets/abides_markets/oracles/sparse_mean_reverting_oracle.py at pinned
// commit f9cbe51 with oracle_scheduled_jump.patch applied.
//
// THREE THINGS MAKE THIS SUBTLE
//
// 1. The megashock inter-arrival comes from the GLOBAL numpy stream (`np.random.exponential`),
//    not the symbol's own RandomState. One draw in the constructor, one per megashock crossing
//    thereafter, interleaved with the simulation. See numpy_legacy_rng.hpp and the contract doc.
//
// 2. PYTHON'S NUMERIC TOWER IS LOAD-BEARING. `mst = self.mkt_open + ms_time_delta` is
//    int + float = float, and nanosecond timestamps are ~1.6e18, well past the 2^53 limit where
//    a double represents every integer. So:
//      - the initial `pt` is an EXACT int (mkt_open), and `d = ts - pt` is an exact int subtraction
//      - once a megashock fires, `pt` becomes a float and `d` becomes a float subtraction
//      - `mst < current_time` is float-vs-int, which CPython compares EXACTLY (it does not
//        coerce the int to double)
//    Doing all of this in double would quantise timestamps to 256 ns and diverge. PyNum below
//    reproduces the tower.
//
// 3. The draw count is data dependent: `if current_time <= pt: return pv` draws nothing, and the
//    megashock `while` loop draws from two different streams per iteration. Only ValueTrader
//    calls observe_price, so a scenario without ValueTraders never advances the fundamental at
//    all.
//
// Megashocks are live in 13 of the 65 public units, including all six reactive-agent units,
// which are Tier A. This path is not an edge case.

#ifndef ORACLE_HPP
#define ORACLE_HPP

#include <cmath>
#include <cstdint>
#include <vector>

#include "numpy_legacy_rng.hpp"

namespace t3 {

// --------------------------------------------------------------------------- numeric tower
// Either an exact Python int or a Python float, with Python's mixed-mode semantics.
struct PyNum {
    bool is_int = true;
    std::int64_t i = 0;
    double d = 0.0;

    static PyNum from_int(std::int64_t v) {
        PyNum n;
        n.is_int = true;
        n.i = v;
        return n;
    }
    static PyNum from_double(double v) {
        PyNum n;
        n.is_int = false;
        n.d = v;
        return n;
    }
    double as_double() const { return is_int ? static_cast<double>(i) : d; }
};

// Exact `a < b` for a double against an int64, without coercing the int to double.
inline bool lt_double_int(double a, std::int64_t b) {
    if (std::isnan(a)) return false;
    if (a < -9.3e18) return true;
    if (a >= 9.3e18) return false;
    const double fa = std::floor(a);
    const std::int64_t ia = static_cast<std::int64_t>(fa);
    // a lies in [ia, ia+1), so a < b iff ia < b.
    return ia < b;
}

// Exact `a >= b` for a double against an int64.
inline bool ge_double_int(double a, std::int64_t b) { return !lt_double_int(a, b); }

// Python `a - b`, both of which may be int or float.
inline double py_sub(const PyNum& a, const PyNum& b) {
    if (a.is_int && b.is_int) {
        return static_cast<double>(a.i - b.i);  // exact int difference first
    }
    return a.as_double() - b.as_double();
}

// Python `a < b` where a is PyNum and b is an exact int.
inline bool py_lt_int(const PyNum& a, std::int64_t b) {
    return a.is_int ? (a.i < b) : lt_double_int(a.d, b);
}

// Python `a <= b` where a is an exact int and b is PyNum (used by `current_time <= pt`).
inline bool int_le_py(std::int64_t a, const PyNum& b) {
    if (b.is_int) return a <= b.i;
    return !lt_double_int(b.d, a);  // a <= b  <=>  not (b < a)
}

// Python `a >= b` where a is PyNum and b is an exact int.
inline bool py_ge_int(const PyNum& a, std::int64_t b) {
    return a.is_int ? (a.i >= b) : ge_double_int(a.d, b);
}

// --------------------------------------------------------------------------- params
struct ScheduledJump {
    std::int64_t time_ns = 0;   // absolute kernel clock (config.py adds mkt_open)
    std::int64_t magnitude = 0;
    bool consumed = false;
};

struct SymbolParams {
    double r_bar = 100000.0;             // also the starting fundamental, an exact int
    double kappa = 1.67e-16;             // per NANOSECOND
    double fund_vol = 5e-5;              // theta
    double megashock_lambda_a = 2.77778e-18;
    double megashock_mean = 1000.0;
    double megashock_var = 50000.0;
    std::vector<ScheduledJump> scheduled_jumps;
};

// --------------------------------------------------------------------------- oracle
class SparseMeanRevertingOracle {
public:
    SparseMeanRevertingOracle(std::int64_t mkt_open, std::int64_t mkt_close, SymbolParams params,
                              nprng::LegacyRandomState* symbol_rs,
                              nprng::LegacyRandomState* global_rs)
        : mkt_open_(mkt_open),
          mkt_close_(mkt_close),
          p_(std::move(params)),
          rs_(symbol_rs),
          global_(global_rs),
          pt_(PyNum::from_int(mkt_open)),
          pv_(static_cast<std::int64_t>(p_.r_bar)) {
        // __init__: r[symbol] = (mkt_open, r_bar); then the first megashock.
        // ms_time_delta comes from the GLOBAL stream; mst = int + float = float.
        const double ms_time_delta = global_->exponential(1.0 / p_.megashock_lambda_a);
        mst_ = PyNum::from_double(static_cast<double>(mkt_open_) + ms_time_delta);
        double msv = rs_->normal(p_.megashock_mean, std::sqrt(p_.megashock_var));
        msv_ = (rs_->randint(0, 2) == 0) ? msv : -msv;
    }

    // advance_fundamental_value_series
    std::int64_t advance(std::int64_t current_time) {
        if (int_le_py(current_time, pt_)) {
            return pv_;  // draws nothing
        }
        while (py_lt_int(mst_, current_time)) {
            const std::int64_t v = compute_at(mst_, msv_, pt_, pv_);
            pt_ = mst_;
            pv_ = v;
            // mst = pt + int(exponential)  -- int() TRUNCATES, and pt is a float here.
            const double delta = global_->exponential(1.0 / p_.megashock_lambda_a);
            const double trunc_delta = std::trunc(delta);
            mst_ = PyNum::from_double(pt_.as_double() + trunc_delta);
            double msv = rs_->normal(p_.megashock_mean, std::sqrt(p_.megashock_var));
            msv_ = (rs_->randint(0, 2) == 0) ? msv : -msv;
        }
        return compute_at(PyNum::from_int(current_time), 0.0, pt_, pv_);
    }

    // observe_price
    std::int64_t observe_price(std::int64_t current_time, nprng::LegacyRandomState& agent_rs,
                               double sigma_n) {
        const std::int64_t r_t = (current_time >= mkt_close_) ? advance(mkt_close_ - 1)
                                                              : advance(current_time);
        if (sigma_n == 0.0) return r_t;
        return nprng::py_round_to_i64(
            agent_rs.normal(static_cast<double>(r_t), std::sqrt(sigma_n)));
    }

    std::int64_t get_daily_open_price() const { return static_cast<std::int64_t>(p_.r_bar); }

private:
    // compute_fundamental_at_timestamp. Updates the cached (pt, pv) as a side effect, exactly
    // as the original does via self.r[symbol].
    std::int64_t compute_at(const PyNum& ts, double v_adj, const PyNum& pt, std::int64_t pv) {
        const double d = py_sub(ts, pt);
        const double mu = p_.r_bar;
        const double gamma = p_.kappa;
        const double theta = p_.fund_vol;

        const double loc = mu + (static_cast<double>(pv) - mu) * std::exp(-gamma * d);
        const double scale =
            std::sqrt(((theta * theta) / (2.0 * gamma)) * (1.0 - std::exp(-2.0 * gamma * d)));
        double v = rs_->normal(loc, scale);
        v += v_adj;
        if (v < 0.0) v = 0.0;                       // max(0, v)
        std::int64_t iv = nprng::py_round_to_i64(v);  // int(round(v)), banker's rounding

        // oracle_scheduled_jump.patch: deterministic, RNG-neutral, applied once.
        for (auto& jump : p_.scheduled_jumps) {
            if (!jump.consumed && py_ge_int(ts, jump.time_ns)) {
                iv = iv + jump.magnitude;
                if (iv < 0) iv = 0;
                jump.consumed = true;
            }
        }

        pt_ = ts;
        pv_ = iv;
        return iv;
    }

    std::int64_t mkt_open_;
    std::int64_t mkt_close_;
    SymbolParams p_;
    nprng::LegacyRandomState* rs_;      // per-symbol stream
    nprng::LegacyRandomState* global_;  // the GLOBAL numpy stream
    PyNum pt_;
    std::int64_t pv_;
    PyNum mst_;
    double msv_ = 0.0;
};

}  // namespace t3

#endif  // ORACLE_HPP
