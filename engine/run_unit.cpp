// run_unit.cpp — run any scenario.json and emit its trace or message ledger.
//
//     ./run_unit <scenario.json> --trace     # 7 trace columns
//     ./run_unit <scenario.json> --ledger    # 10 message_trace columns
//
// "-" encodes a null. This is the shape the eventual `simulate` CLI will take, minus the Parquet
// writer (which has to be compiled, since importing pyarrow costs ~500 ms == 4.27x on the score).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "scenario.hpp"
#include "sim.hpp"

using namespace t3;

// --bench N: run the scenario N times in-process and report timing. Excludes process start-up
// so the number is comparable to the baseline's self-reported events.json wall_clock_sec and to
// the Development clip. It is NOT the Final-path number, which is whole-container wall clock.
static int bench(const std::string& path, int repeats, bool build_outputs) {
    const auto p0 = std::chrono::steady_clock::now();
    const LoadedScenario sc = load_scenario(path);
    const auto p1 = std::chrono::steady_clock::now();
    const double parse_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

    std::vector<double> ms;
    ms.reserve(static_cast<std::size_t>(repeats));
    std::size_t n_trace = 0, n_msgs = 0;
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const SimResult res = run_simulation(sc.cfg, build_outputs);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        n_trace = res.trace.size();
        n_msgs = res.n_messages;
    }
    std::sort(ms.begin(), ms.end());
    const double best = ms.front();
    const double median = ms[ms.size() / 2];
    // n_trace n_msgs best_ms median_ms parse_ms
    std::printf("%zu %zu %.6f %.6f %.6f\n", n_trace, n_msgs, best, median, parse_ms);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: run_unit <scenario.json> --trace|--ledger|--bench N [--no-outputs]\n");
        return 2;
    }
    const std::string path = argv[1];

    try {
        if (std::strcmp(argv[2], "--bench") == 0) {
            const int repeats = argc > 3 ? std::atoi(argv[3]) : 5;
            bool build_outputs = true;
            for (int i = 3; i < argc; ++i)
                if (std::strcmp(argv[i], "--no-outputs") == 0) build_outputs = false;
            return bench(path, repeats > 0 ? repeats : 1, build_outputs);
        }

        const bool want_trace = std::strcmp(argv[2], "--trace") == 0;
        const LoadedScenario sc = load_scenario(path);
        const SimResult res = run_simulation(sc.cfg, /*build_outputs=*/true);

        if (want_trace) {
            for (const TraceRow& r : res.trace) {
                std::printf("%lld %d %s %s %lld %lld %lld\n", (long long)r.t_ns, r.agent_id,
                            r.msg_type, r.side[0] ? r.side : "-", (long long)r.price,
                            (long long)r.size, (long long)r.order_id);
            }
        } else {
            for (const LedgerRow& r : res.ledger) {
                char ts[32], oid[32], par[32];
                if (r.has_send) std::snprintf(ts, sizeof ts, "%lld", (long long)r.t_send_ns);
                else std::snprintf(ts, sizeof ts, "-");
                if (r.order_id >= 0) std::snprintf(oid, sizeof oid, "%lld", (long long)r.order_id);
                else std::snprintf(oid, sizeof oid, "-");
                if (r.causal_parent >= 0)
                    std::snprintf(par, sizeof par, "%lld", (long long)r.causal_parent);
                else std::snprintf(par, sizeof par, "-");
                std::printf("%lld %lld %s %lld %d %d %lld %s %s %s\n", (long long)r.seq,
                            (long long)r.t_recv_ns, ts, (long long)r.latency_ns, r.src_id,
                            r.dst_id, (long long)r.message_id, msg_type_name(r.msg_type), oid,
                            par);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "run_unit: %s\n", e.what());
        return 1;
    }
}
