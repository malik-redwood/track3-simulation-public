// bench.cpp — measure the engine's simulation-loop throughput.
//
// WHAT NUMBER THIS IS, AND WHAT IT IS NOT
//
// This times the simulation loop only (plus trace assembly when asked), which is the same
// interval the ABIDES baseline reports in events.json and the same one the Development board
// scores. It is therefore directly comparable to:
//     * the baseline's 14,159 events/sec arithmetic mean over the 65 units
//     * the Development per-unit clip of 1e7 events/sec
//
// It is NOT the Final path number. Final timing is whole-container wall clock
// (State.StartedAt -> State.FinishedAt), measured at ~92.5 ms for a do-nothing static binary,
// which dominates every unit smaller than ~1e6 events.
//
//     ./bench                 # all configs, 5 repeats
//     ./bench --repeats 20
//     ./bench --no-outputs    # exclude trace/ledger assembly, isolating the event loop

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sim.hpp"

using namespace t3;

struct Case {
    const char* name;
    SimConfig cfg;
};

static SimConfig scaled(int n_noise, int n_mm, int n_value, int n_momentum,
                        std::int64_t horizon_ns, double hz) {
    SimConfig c = config_s001();
    c.agents.clear();
    const std::int64_t iv = static_cast<std::int64_t>(1e9 / hz);
    if (n_noise) {
        AgentSpec s;
        s.kind = AgentKind::Noise;
        s.count = n_noise;
        s.interval_ns = iv;
        c.agents.push_back(s);
    }
    if (n_mm) {
        AgentSpec s;
        s.kind = AgentKind::MarketMakerKind;
        s.count = n_mm;
        s.interval_ns = 1000000000LL;
        c.agents.push_back(s);
    }
    if (n_value) {
        AgentSpec s;
        s.kind = AgentKind::Value;
        s.count = n_value;
        s.interval_ns = iv;
        s.order_size_mean = 25.0;
        c.agents.push_back(s);
    }
    if (n_momentum) {
        AgentSpec s;
        s.kind = AgentKind::Momentum;
        s.count = n_momentum;
        s.interval_ns = iv;
        s.order_size_mean = 15.0;
        c.agents.push_back(s);
    }
    c.horizon_ns = horizon_ns;
    return c;
}

int main(int argc, char** argv) {
    int repeats = 5;
    bool build_outputs = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-outputs") == 0) build_outputs = false;
    }

    std::vector<Case> cases = {
        {"t3-s001 (3 noise + 1 mm, 10s)", config_s001()},
        {"20 agents, 30s @2Hz", scaled(14, 2, 2, 2, 30000000000LL, 2.0)},
        {"30 agents, 30s @2Hz", scaled(20, 4, 3, 3, 30000000000LL, 2.0)},
        {"50 agents, 60s @40Hz", scaled(34, 6, 5, 5, 60000000000LL, 40.0)},
        {"128 agents, 30s @2Hz", scaled(100, 10, 9, 9, 30000000000LL, 2.0)},
        {"30 agents, 240s @2Hz", scaled(20, 4, 3, 3, 240000000000LL, 2.0)},
    };

    std::printf("outputs=%s  repeats=%d\n\n", build_outputs ? "trace+ledger" : "event loop only",
                repeats);
    std::printf("%-32s %10s %10s %12s %14s %12s\n", "config", "trace", "msgs", "best_ms",
                "events/sec", "msgs/sec");
    std::printf("%s\n", std::string(96, '-').c_str());

    for (Case& c : cases) {
        double best = 1e30;
        std::size_t n_trace = 0, n_msg = 0;
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            SimResult res = run_simulation(c.cfg, build_outputs);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
            n_trace = res.trace.size();
            n_msg = res.n_messages;
        }
        // events/sec uses the TRACE row count, which is what the harness counts and ranks.
        const double eps = build_outputs ? (static_cast<double>(n_trace) / (best / 1000.0)) : 0.0;
        const double mps = static_cast<double>(n_msg) / (best / 1000.0);
        if (build_outputs) {
            std::printf("%-32s %10zu %10zu %12.3f %14.0f %12.0f\n", c.name, n_trace, n_msg, best,
                        eps, mps);
        } else {
            std::printf("%-32s %10s %10zu %12.3f %14s %12.0f\n", c.name, "-", n_msg, best, "-",
                        mps);
        }
    }
    std::printf("\nevents/sec counts TRACE rows (the ranked numerator). msgs/sec counts kernel\n"
                "messages, which is the better measure of raw event-loop speed.\n");
    return 0;
}
