// test_rng.cpp — run RNG cases from a spec on stdin, emit results for exact comparison.
//
// Doubles are emitted as their raw IEEE-754 bit pattern, so the comparison in check_rng.py is
// bit-exact rather than tolerance-based. A tolerance would hide precisely the kind of
// one-word-off error this test exists to catch.
//
// Spec lines (one case each), whitespace separated:
//   raw_words      <seed> <n>
//   random_sample  <seed> <n>
//   normal         <seed> <n> <loc> <scale>
//   uniform        <seed> <n> <low> <high>
//   exponential    <seed> <n> <scale>
//   pareto         <seed> <n> <a>
//   lognormal      <seed> <n> <mean> <sigma>
//   randint_small  <seed> <n> <low> <high>
//   global_chain   <scenario_seed> <n_agents> <lambda>
//   noise_trader   <seed> <n> <mean> <std> <offset_ticks>
//   latency        <seed> <n> <model> <mean_ns> <sigma> <min_ns> <max_ns> <alpha>
//
// Output: "<case_idx> <item_idx> <field> <value>", doubles as 16 hex digits.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "numpy_legacy_rng.hpp"
#include "oracle.hpp"

using nprng::clip;
using nprng::LegacyRandomState;
using nprng::py_round_to_i64;

static int g_case = -1;

static void emit_d(int item, const char* field, double v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    std::printf("%d %d %s %016llx\n", g_case, item, field,
                static_cast<unsigned long long>(bits));
}

static void emit_i(int item, const char* field, long long v) {
    std::printf("%d %d %s %lld\n", g_case, item, field, v);
}

int main() {
    std::string kind;
    while (std::cin >> kind) {
        ++g_case;

        if (kind == "raw_words") {
            std::uint64_t seed;
            int n;
            std::cin >> seed >> n;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_i(i, "value", static_cast<long long>(rs.draw_seed()));
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "random_sample") {
            std::uint64_t seed;
            int n;
            std::cin >> seed >> n;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_d(i, "value", rs.random_sample());
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "normal") {
            std::uint64_t seed;
            int n;
            double loc, scale;
            std::cin >> seed >> n >> loc >> scale;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_d(i, "value", rs.normal(loc, scale));
                emit_i(i, "pos", rs.word_pos());
                emit_i(i, "has_gauss", rs.has_gauss() ? 1 : 0);
            }

        } else if (kind == "uniform") {
            std::uint64_t seed;
            int n;
            double lo, hi;
            std::cin >> seed >> n >> lo >> hi;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_d(i, "value", rs.uniform(lo, hi));
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "exponential") {
            std::uint64_t seed;
            int n;
            double scale;
            std::cin >> seed >> n >> scale;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_d(i, "value", rs.exponential(scale));
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "pareto") {
            std::uint64_t seed;
            int n;
            double a;
            std::cin >> seed >> n >> a;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_d(i, "value", rs.pareto(a));
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "lognormal") {
            std::uint64_t seed;
            int n;
            double mean, sigma;
            std::cin >> seed >> n >> mean >> sigma;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_d(i, "value", rs.lognormal(mean, sigma));
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "randint_small") {
            std::uint64_t seed;
            int n;
            long long lo, hi;
            std::cin >> seed >> n >> lo >> hi;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                emit_i(i, "value", static_cast<long long>(rs.randint(lo, hi)));
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "global_chain") {
            // The verified abides_fork/config.py draw order. The oracle's first megashock
            // inter-arrival comes from the GLOBAL stream (np.random.exponential) and sits
            // between the oracle's own sub-seed and the exchange's -- miss it and every agent
            // gets the wrong seed.
            std::uint64_t scenario_seed;
            int n_agents;
            double lambda;
            std::cin >> scenario_seed >> n_agents >> lambda;
            LegacyRandomState g(static_cast<std::uint32_t>(scenario_seed));
            emit_i(0, "oracle_seed", static_cast<long long>(g.draw_seed()));
            emit_d(0, "first_megashock_delta", g.exponential(1.0 / lambda));
            emit_i(0, "exchange_seed", static_cast<long long>(g.draw_seed()));
            for (int i = 0; i < n_agents; ++i) {
                emit_i(i, "agent_seed", static_cast<long long>(g.draw_seed()));
            }
            emit_i(0, "latency_seed", static_cast<long long>(g.draw_seed()));
            emit_i(0, "kernel_seed", static_cast<long long>(g.draw_seed()));

        } else if (kind == "noise_trader") {
            // agents.py NoiseTrader.act(): exactly three draws, in this order.
            std::uint64_t seed;
            int n, offset_ticks;
            double mean, sd;
            std::cin >> seed >> n >> mean >> sd >> offset_ticks;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                const long long size =
                    std::max<long long>(1, py_round_to_i64(rs.normal(mean, sd)));
                const long long buy = rs.randint(0, 2);
                const long long off = rs.randint(0, offset_ticks + 1);
                emit_i(i, "size", size);
                emit_i(i, "buy", buy != 0 ? 1 : 0);
                emit_i(i, "offset", off);
                emit_i(i, "pos", rs.word_pos());
            }

        } else if (kind == "latency") {
            // config.py ScenarioLatencyModel.get_latency for sender != recipient.
            std::uint64_t seed;
            int n;
            std::string model;
            double mean_ns, sigma, min_ns, max_ns, alpha;
            std::cin >> seed >> n >> model >> mean_ns >> sigma >> min_ns >> max_ns >> alpha;
            const double mu = mean_ns > 0.0 ? std::log(mean_ns) : 0.0;
            LegacyRandomState rs(static_cast<std::uint32_t>(seed));
            for (int i = 0; i < n; ++i) {
                double v;
                if (model == "log_normal") {
                    v = rs.lognormal(mu, sigma);
                } else if (model == "uniform") {
                    v = rs.uniform(min_ns, max_ns);
                } else if (model == "pareto") {
                    v = (min_ns > 0.0 ? min_ns : 1.0) * (1.0 + rs.pareto(alpha));
                } else {
                    v = mean_ns;
                }
                emit_i(i, "value", py_round_to_i64(clip(v, min_ns, max_ns)));
            }

        } else if (kind == "oracle") {
            // Mirrors config.py's construction order: the global stream is seeded from the
            // scenario seed, the oracle's own sub-seed is drawn from it (word 1), and then the
            // oracle constructor consumes the SAME global stream for its first megashock.
            std::uint64_t global_seed, agent_seed;
            double r_bar, kappa_per_s, sigma, jump_intensity, jump_sigma, sigma_n;
            long long horizon_ns, jump_time_ns, jump_mag;
            int n_obs;
            std::cin >> global_seed >> agent_seed >> r_bar >> kappa_per_s >> sigma
                     >> jump_intensity >> jump_sigma >> horizon_ns >> sigma_n >> n_obs
                     >> jump_time_ns >> jump_mag;

            // config.py constants and conversions.
            const std::int64_t date_ns = 1612483200000000000LL;       // pd.to_datetime("20210205")
            const std::int64_t mkt_open = date_ns + 34200000000000LL;  // + 09:30:00
            const std::int64_t mkt_close = mkt_open + horizon_ns;

            t3::SymbolParams p;
            p.r_bar = r_bar;
            p.kappa = kappa_per_s > 0.0 ? kappa_per_s / 1e9 : 1.67e-16;
            p.fund_vol = sigma;
            p.megashock_lambda_a = jump_intensity > 0.0 ? jump_intensity / 1e9 : 2.77778e-18;
            // `float(jump_sigma) or 1000.0` -- note 0.0 is falsy in Python, so it becomes 1000.
            p.megashock_mean = (jump_sigma != 0.0) ? jump_sigma : 1000.0;
            p.megashock_var = 50000.0;
            if (jump_time_ns >= 0) {
                t3::ScheduledJump j;
                j.time_ns = mkt_open + jump_time_ns;
                j.magnitude = jump_mag;
                p.scheduled_jumps.push_back(j);
            }

            nprng::LegacyRandomState global(static_cast<std::uint32_t>(global_seed));
            nprng::LegacyRandomState symbol_rs(global.draw_seed());
            t3::SparseMeanRevertingOracle oracle(mkt_open, mkt_close, p, &symbol_rs, &global);
            nprng::LegacyRandomState agent(static_cast<std::uint32_t>(agent_seed));

            for (int i = 0; i < n_obs; ++i) {
                const std::int64_t t =
                    mkt_open + static_cast<std::int64_t>((i + 1)) * horizon_ns / n_obs;
                emit_i(i, "obs", static_cast<long long>(oracle.observe_price(t, agent, sigma_n)));
            }

        } else {
            std::fprintf(stderr, "unknown case kind: %s\n", kind.c_str());
            return 2;
        }
    }
    return 0;
}
