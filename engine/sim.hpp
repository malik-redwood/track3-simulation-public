// sim.hpp — assemble and run one scenario, mirroring abides_fork/config.py::build_config.
//
// The global draw order below is the verified chain (65/65 units):
//     word 1          oracle symbol seed
//     words 2,3       np.random.exponential  (oracle ctor, GLOBAL stream)
//     word 4          ExchangeAgent seed     (drawn, never consumed)
//     words 5..4+N    agent seeds, construction order == agent_id 1..N
//     word 5+N        latency model seed
//     word 6+N        kernel seed            (unused when latency_config is present)

#ifndef SIM_HPP
#define SIM_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "agents.hpp"
#include "exchange_agent.hpp"
#include "kernel.hpp"
#include "oracle.hpp"
#include "trace.hpp"

namespace t3 {

enum class AgentKind { Noise, MarketMakerKind, Value, Momentum };

struct AgentSpec {
    AgentKind kind = AgentKind::Noise;
    int count = 1;
    std::int64_t interval_ns = 500000000;  // rebalance_interval_ns, else 1e9/arrival_rate_hz
    // NoiseTrader
    double order_size_mean = 10.0;
    double order_size_std = 2.0;
    int price_offset_ticks = 5;
    // MarketMaker
    int spread_ticks = 2;
    int depth_levels = 3;
    std::int64_t size_per_level = 10;
    // Value / Momentum
    int threshold_ticks = 2;
    double sigma_n = 1000.0;
    int lookback = 5;
};

struct SimConfig {
    std::uint32_t seed = 1001;
    std::int64_t horizon_ns = 10000000000LL;
    std::int64_t reference_price = 100000;
    // oracle
    double kappa_per_s = 0.05;
    double sigma = 0.0005;
    double jump_intensity = 0.0;
    double jump_sigma = 0.0;
    bool has_scheduled_jump = false;
    std::int64_t jump_time_ns_rel = 0;  // relative to mkt_open
    std::int64_t jump_magnitude = 0;
    // Exchange protocol, opt-in via exchange_config.protocol_enforcement. Empty policy == off.
    std::string stp_policy;
    int ack_delay_ns = 0;
    int compute_delay_ns = 0;
    // latency
    LatencyModel::Kind latency_kind = LatencyModel::Kind::Uniform;
    double latency_mean_ns = 500.0, latency_sigma = 0.0;
    double latency_min_ns = 100.0, latency_max_ns = 2000.0, latency_alpha = 1.5;
    std::vector<AgentSpec> agents;
};

struct SimResult {
    std::vector<LedgerRow> ledger;
    std::vector<TraceRow> trace;
    std::size_t n_messages = 0;
};

// Runs one scenario to completion. Resets the global id counters first, exactly as
// simulate.py::reset_abides_counters does, so repeated in-process runs are independent.
inline SimResult run_simulation(const SimConfig& cfg, bool build_outputs = true) {
    reset_abides_counters();

    const std::int64_t date_ns = 1612483200000000000LL;       // pd.to_datetime("20210205")
    const std::int64_t mkt_open = date_ns + 34200000000000LL;  // + 09:30:00
    const std::int64_t mkt_close = mkt_open + cfg.horizon_ns;
    const std::int64_t oracle_close = date_ns + 57600000000000LL;  // + 16:00:00
    const std::int64_t stop_time = mkt_close + 1000000000LL;       // + 1s

    int n_agents = 0;
    for (const AgentSpec& s : cfg.agents) n_agents += s.count;

    nprng::LegacyRandomState global(cfg.seed);
    nprng::LegacyRandomState oracle_rs(global.draw_seed());

    SymbolParams sp;
    sp.r_bar = static_cast<double>(cfg.reference_price);
    sp.kappa = cfg.kappa_per_s > 0.0 ? cfg.kappa_per_s / 1e9 : 1.67e-16;
    sp.fund_vol = cfg.sigma;
    sp.megashock_lambda_a = cfg.jump_intensity > 0.0 ? cfg.jump_intensity / 1e9 : 2.77778e-18;
    sp.megashock_mean = (cfg.jump_sigma != 0.0) ? cfg.jump_sigma : 1000.0;
    sp.megashock_var = 50000.0;
    if (cfg.has_scheduled_jump) {
        ScheduledJump j;
        j.time_ns = mkt_open + cfg.jump_time_ns_rel;  // config.py converts to absolute
        j.magnitude = cfg.jump_magnitude;
        sp.scheduled_jumps.push_back(j);
    }
    SparseMeanRevertingOracle oracle(mkt_open, oracle_close, sp, &oracle_rs, &global);

    nprng::LegacyRandomState exchange_rs(global.draw_seed());
    std::vector<nprng::LegacyRandomState> rs;
    rs.reserve(static_cast<std::size_t>(n_agents));
    for (int i = 0; i < n_agents; ++i) rs.emplace_back(global.draw_seed());
    const std::uint32_t latency_seed = global.draw_seed();
    (void)global.draw_seed();  // kernel seed: drawn, unused

    Kernel kernel(date_ns, stop_time, /*default_computation_delay=*/50);
    kernel.set_latency_model(LatencyModel(cfg.latency_kind, cfg.latency_mean_ns,
                                          cfg.latency_sigma, cfg.latency_min_ns,
                                          cfg.latency_max_ns, cfg.latency_alpha, latency_seed));

    ExchangeAgent exchange(0, exchange_rs, mkt_open, mkt_close, oracle.get_daily_open_price(),
                           cfg.compute_delay_ns, cfg.ack_delay_ns, cfg.stp_policy);
    kernel.add_agent(&exchange);

    std::vector<std::unique_ptr<ScheduledAgent>> traders;
    int next_id = 1;
    std::size_t rs_i = 0;
    for (const AgentSpec& s : cfg.agents) {
        for (int k = 0; k < s.count; ++k) {
            const int id = next_id++;
            nprng::LegacyRandomState r = rs[rs_i++];
            switch (s.kind) {
                case AgentKind::Noise:
                    traders.push_back(std::make_unique<NoiseTrader>(
                        id, r, 0, s.interval_ns, s.order_size_mean, s.order_size_std,
                        s.price_offset_ticks, cfg.reference_price));
                    break;
                case AgentKind::MarketMakerKind:
                    traders.push_back(std::make_unique<MarketMaker>(
                        id, r, 0, s.interval_ns, s.spread_ticks, s.depth_levels,
                        s.size_per_level, cfg.reference_price));
                    break;
                case AgentKind::Value:
                    traders.push_back(std::make_unique<ValueTrader>(
                        id, r, 0, s.interval_ns, s.order_size_mean, s.threshold_ticks,
                        s.sigma_n, &oracle));
                    break;
                case AgentKind::Momentum:
                    traders.push_back(std::make_unique<MomentumTrader>(
                        id, r, 0, s.interval_ns, s.order_size_mean, s.threshold_ticks,
                        s.lookback));
                    break;
            }
            kernel.add_agent(traders.back().get());
        }
    }

    kernel.run();

    SimResult out;
    out.n_messages = kernel.ledger().size();
    if (build_outputs) {
        out.ledger = kernel.delivered_ledger_sorted();
        std::vector<LogEvent> events;
        for (const LogEvent& e : exchange.log()) events.push_back(e);
        for (auto& t : traders)
            for (const LogEvent& e : t->log()) events.push_back(e);
        out.trace = extract_trace(events);
    }
    return out;
}

// The t3-s001-price-time-priority scenario, as a known-good baseline config.
inline SimConfig config_s001() {
    SimConfig c;
    c.seed = 1001;
    c.horizon_ns = 10000000000LL;
    c.reference_price = 100000;
    c.kappa_per_s = 0.05;
    c.sigma = 0.0005;
    c.latency_kind = LatencyModel::Kind::Uniform;
    c.latency_min_ns = 100.0;
    c.latency_max_ns = 2000.0;
    c.latency_mean_ns = 500.0;
    AgentSpec noise;
    noise.kind = AgentKind::Noise;
    noise.count = 3;
    noise.interval_ns = static_cast<std::int64_t>(1e9 / 2.0);
    AgentSpec mm;
    mm.kind = AgentKind::MarketMakerKind;
    mm.count = 1;
    mm.interval_ns = 1000000000LL;
    c.agents = {noise, mm};
    return c;
}

}  // namespace t3

#endif  // SIM_HPP
