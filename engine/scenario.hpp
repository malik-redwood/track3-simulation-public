// scenario.hpp — map a Track-3 scenario.json onto SimConfig, mirroring
// abides_fork/config.py::build_config.
//
// THE MAPPINGS THAT ARE NOT OBVIOUS FROM THE JSON
//
//  * kappa and jump_intensity are given PER SECOND and divided by 1e9. Absent or zero falls back
//    to the gentle rmsc04 defaults (1.67e-16 and 2.77778e-18), which make megashocks effectively
//    never fire.
//  * megashock_mean = `float(jump_sigma) or 1000.0` -- 0.0 is FALSY in Python, so a configured
//    zero silently becomes 1000.0.
//  * interval_ns = rebalance_interval_ns if present, else int(1e9 / arrival_rate_hz).
//  * stp_policy / ack_delay_ns / compute_delay_ns are OPT-IN: ignored entirely unless
//    exchange_config.protocol_enforcement is true. 7 of 65 units set it. Otherwise the exchange
//    runs with stp_policy = None and both delays 0.
//  * latency `scale_ns` is never read by config.py (pareto scales off min_ns), and the one unit
//    carrying it has no mean_ns at all.
//  * scheduled_jump.time_ns is RELATIVE to mkt_open; config.py converts to the absolute clock.
//  * exchange_config.symbol / tick_size / lot_size / min_price / max_price / order_types_allowed
//    are not consumed by the baseline at all -- the symbol is cosmetic and the rest unused.

#ifndef SCENARIO_HPP
#define SCENARIO_HPP

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "json.hpp"
#include "sim.hpp"

namespace t3 {

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline LatencyModel::Kind latency_kind_from(const std::string& model) {
    if (model == "log_normal") return LatencyModel::Kind::LogNormal;
    if (model == "uniform") return LatencyModel::Kind::Uniform;
    if (model == "pareto") return LatencyModel::Kind::Pareto;
    return LatencyModel::Kind::Deterministic;
}

struct LoadedScenario {
    SimConfig cfg;
    std::string scenario_id;
};

inline LoadedScenario load_scenario(const std::string& path) {
    const minijson::Value root = minijson::parse(read_file(path));
    LoadedScenario out;
    out.scenario_id = root.get_string("scenario_id", "");

    SimConfig& c = out.cfg;
    c.seed = static_cast<std::uint32_t>(root.get_int("seed", 0));
    c.horizon_ns = root.get_int("horizon_ns", 0);

    // ---- oracle -------------------------------------------------------
    const minijson::Value& oc = root.at("oracle_config");
    static const minijson::Value kEmpty;
    const minijson::Value& op = oc.has("params") ? oc.at("params") : kEmpty;
    c.reference_price = op.get_int("initial_price", 100000);
    c.kappa_per_s = op.get_double("kappa", 0.0);
    c.sigma = op.get_double("sigma", 5e-5);
    c.jump_intensity = op.get_double("jump_intensity", 0.0);
    c.jump_sigma = op.get_double("jump_sigma", 0.0);
    if (op.has("scheduled_jump")) {
        const minijson::Value& sj = op.at("scheduled_jump");
        c.has_scheduled_jump = true;
        c.jump_time_ns_rel = sj.get_int("time_ns", 0);
        c.jump_magnitude = sj.get_int("magnitude", 0);
    }

    // ---- exchange protocol (opt-in) -----------------------------------
    const minijson::Value& ec = root.at("exchange_config");
    if (ec.get_bool("protocol_enforcement", false)) {
        c.stp_policy = ec.get_string("stp_policy", "");
        c.ack_delay_ns = static_cast<int>(ec.get_int("ack_delay_ns", 0));
        c.compute_delay_ns = static_cast<int>(ec.get_int("compute_delay_ns", 0));
    }

    // ---- latency ------------------------------------------------------
    const minijson::Value& lc = root.at("latency_config");
    const minijson::Value& lp = lc.has("params") ? lc.at("params") : kEmpty;
    c.latency_kind = latency_kind_from(lc.get_string("model", "deterministic"));
    c.latency_mean_ns = lp.get_double("mean_ns", 0.0);
    c.latency_sigma = lp.get_double("sigma", 0.0);
    c.latency_min_ns = lp.get_double("min_ns", 0.0);
    c.latency_max_ns = lp.get_double("max_ns", 1e12);
    c.latency_alpha = lp.get_double("alpha", 1.5);

    // ---- agents (construction order fixes agent ids and seed order) ---
    c.agents.clear();
    for (const minijson::Value& a : root.at("agent_configs").arr) {
        const std::string type = a.get_string("agent_type", "");
        const minijson::Value& p = a.has("params") ? a.at("params") : kEmpty;
        AgentSpec s;
        s.count = static_cast<int>(a.get_int("count", 0));
        if (p.has("rebalance_interval_ns")) {
            s.interval_ns = p.get_int("rebalance_interval_ns", 1000000000LL);
        } else {
            const double hz = p.get_double("arrival_rate_hz", 1.0);
            s.interval_ns = hz > 0.0 ? static_cast<std::int64_t>(1e9 / hz)
                                     : static_cast<std::int64_t>(1e9);
        }
        if (type == "NoiseTrader") {
            s.kind = AgentKind::Noise;
            s.order_size_mean = p.get_double("order_size_mean", 10.0);
            s.order_size_std = p.get_double("order_size_std", 2.0);
            s.price_offset_ticks = static_cast<int>(p.get_int("price_offset_ticks", 5));
        } else if (type == "MarketMaker") {
            s.kind = AgentKind::MarketMakerKind;
            s.spread_ticks = static_cast<int>(p.get_int("spread_ticks", 2));
            s.depth_levels = static_cast<int>(p.get_int("depth_levels", 3));
            s.size_per_level = p.get_int("size_per_level", 10);
        } else if (type == "ValueTrader") {
            s.kind = AgentKind::Value;
            s.order_size_mean = p.get_double("order_size_mean", 25.0);
            s.threshold_ticks = static_cast<int>(p.get_int("threshold_ticks", 2));
            s.sigma_n = 1000.0;  // agents.py default; no scenario overrides it
        } else if (type == "MomentumTrader") {
            s.kind = AgentKind::Momentum;
            s.order_size_mean = p.get_double("order_size_mean", 15.0);
            s.threshold_ticks = static_cast<int>(p.get_int("threshold_ticks", 2));
            s.lookback = static_cast<int>(p.get_int("lookback", 5));
        } else {
            throw std::runtime_error("unsupported agent_type: " + type);
        }
        c.agents.push_back(s);
    }
    return out;
}

}  // namespace t3

#endif  // SCENARIO_HPP
