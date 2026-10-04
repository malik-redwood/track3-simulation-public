// simulate.cpp — THE SUBMISSION ENTRY POINT.
//
//     simulate       --config /input/scenario.json --out /output/trace.parquet [--seed N]
//     simulate-batch --batch-dir /input/scenarios   --out-dir /output
//
// The verb arrives as the container command, i.e. the first argument, so it is accepted both as
// argv[1] and implicitly (a lone --config means `simulate`).
//
// Outputs, per baselines/README.md "Interface contract":
//   simulate       -> <out>, message_trace.parquet and events.json BESIDE it
//   simulate-batch -> <out-dir>/<sub>/{trace,message_trace}.parquet + events.json per sub,
//                     plus <out-dir>/batch_events.json
//
// events.json carries all six required keys (scenario_id, n_events, wall_clock_sec,
// events_per_sec, seed, trace_sha256) plus the two telemetry keys. n_events MUST equal the real
// trace row count -- the harness counts rows itself and a mismatch fails the run.
//
// batch_events.json MUST include wall_clock_sec: check_aggregate reads total_events,
// wall_clock_sec and events_per_sec together, and a missing key fails the whole batch with
// "non-numeric batch_events fields", a message that does not name the field it wanted.
//
// No Python, no numpy, no pyarrow: importing that stack costs ~500 ms of container start-up,
// which is 4.27x on the ranking score.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The submission image is FROM scratch: there is no /bin/sh, so directory creation and
// listing must use POSIX calls directly. std::system() and popen() would both fail there.
#if defined(__unix__) || defined(__APPLE__)
#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <direct.h>
#include <io.h>
#endif

#include <algorithm>

#include "parquet.hpp"
#include "scenario.hpp"
#include "sha256.hpp"
#include "sim.hpp"

using namespace t3;

// ---------------------------------------------------------------- small helpers
static std::string dir_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

static std::int64_t peak_rss_bytes() {
#if defined(__unix__)
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        // Linux reports ru_maxrss in KiB, which is what simulate.py assumes.
        return static_cast<std::int64_t>(ru.ru_maxrss) * 1024;
    }
#endif
    return 0;
}

// Create a directory and any missing parents. No shell: a scratch image has none.
static void mkdir_p(const std::string& path) {
    if (path.empty() || path == "." || path == "/") return;
    std::string acc;
    std::size_t i = 0;
    if (path[0] == '/') { acc = "/"; i = 1; }
    while (i <= path.size()) {
        const std::size_t slash = path.find_first_of("/\\", i);
        const std::size_t end = (slash == std::string::npos) ? path.size() : slash;
        const std::string part = path.substr(i, end - i);
        if (!part.empty()) {
            if (!acc.empty() && acc.back() != '/') acc += "/";
            acc += part;
#if defined(_WIN32)
            _mkdir(acc.c_str());
#else
            ::mkdir(acc.c_str(), 0755);  // EEXIST is fine
#endif
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
}

// List *.json in a directory, sorted -- the order simulate_batch.py iterates.
static std::vector<std::string> list_json(const std::string& dir) {
    std::vector<std::string> out;
#if defined(__unix__) || defined(__APPLE__)
    if (DIR* d = ::opendir(dir.c_str())) {
        while (struct dirent* e = ::readdir(d)) {
            const std::string name = e->d_name;
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".json") == 0) {
                out.push_back(dir + "/" + name);
            }
        }
        ::closedir(d);
    }
#else
    // Dev-box only; the scored path is Linux.
    const std::string pat = dir + "/*.json";
    struct _finddata_t fd;
    intptr_t h = _findfirst(pat.c_str(), &fd);
    if (h != -1) {
        do {
            out.push_back(dir + "/" + fd.name);
        } while (_findnext(h, &fd) == 0);
        _findclose(h);
    }
#endif
    std::sort(out.begin(), out.end());
    return out;
}

// A double formatted with enough precision to round-trip, for events.json.
static std::string num(double v) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

// ---------------------------------------------------------------- parquet shaping
static void write_trace_parquet(const std::string& path, const std::vector<TraceRow>& rows) {
    std::vector<pq::Column> cols;

    pq::Column t_ns;  t_ns.name = "t_ns";  t_ns.type = pq::Ty::Int64;  t_ns.numpy_type = "int64";
    pq::Column ag;    ag.name = "agent_id"; ag.type = pq::Ty::Int32;   ag.numpy_type = "int32";
    pq::Column mt;    mt.name = "msg_type"; mt.type = pq::Ty::String;  mt.numpy_type = "object";
    // `side` is null on rows that have no side; the reference stores it as a nullable string.
    pq::Column sd;    sd.name = "side";    sd.type = pq::Ty::String;  sd.optional = true;
                      sd.numpy_type = "object";
    pq::Column px;    px.name = "price";   px.type = pq::Ty::Int64;   px.numpy_type = "int64";
    pq::Column sz;    sz.name = "size";    sz.type = pq::Ty::Int64;   sz.numpy_type = "int64";
    pq::Column oid;   oid.name = "order_id"; oid.type = pq::Ty::Int64; oid.numpy_type = "int64";

    t_ns.i64.reserve(rows.size()); ag.i32.reserve(rows.size()); mt.str.reserve(rows.size());
    sd.str.reserve(rows.size());   sd.present.reserve(rows.size());
    px.i64.reserve(rows.size());   sz.i64.reserve(rows.size()); oid.i64.reserve(rows.size());

    for (const TraceRow& r : rows) {
        t_ns.i64.push_back(r.t_ns);
        ag.i32.push_back(r.agent_id);
        mt.str.emplace_back(r.msg_type);
        const bool has_side = r.side && r.side[0] != '\0';
        sd.str.emplace_back(has_side ? r.side : "");
        sd.present.push_back(has_side ? 1 : 0);
        px.i64.push_back(r.price);
        sz.i64.push_back(r.size);
        oid.i64.push_back(r.order_id);
    }
    cols = {t_ns, ag, mt, sd, px, sz, oid};
    pq::write(path, cols);
}

static void write_ledger_parquet(const std::string& path, const std::vector<LedgerRow>& rows) {
    std::vector<pq::Column> cols;

    pq::Column seq;  seq.name = "seq";       seq.type = pq::Ty::Int64; seq.numpy_type = "int64";
    pq::Column trec; trec.name = "t_recv_ns"; trec.type = pq::Ty::Int64; trec.numpy_type = "int64";
    // The three nullable columns MUST be declared Int64 (capital I). Declared as plain int64
    // they come back from pd.read_parquet as float64, and float64 cannot hold a 1.6e18
    // timestamp -- the gate then reports bogus latency-identity failures.
    pq::Column tsnd; tsnd.name = "t_send_ns"; tsnd.type = pq::Ty::Int64; tsnd.optional = true;
                     tsnd.numpy_type = "Int64";
    pq::Column lat;  lat.name = "latency_ns"; lat.type = pq::Ty::Int64; lat.numpy_type = "int64";
    pq::Column src;  src.name = "src_id";    src.type = pq::Ty::Int32; src.numpy_type = "int32";
    pq::Column dst;  dst.name = "dst_id";    dst.type = pq::Ty::Int32; dst.numpy_type = "int32";
    pq::Column mid;  mid.name = "message_id"; mid.type = pq::Ty::Int64; mid.numpy_type = "int64";
    pq::Column mt;   mt.name = "msg_type";   mt.type = pq::Ty::String; mt.numpy_type = "object";
    pq::Column oid;  oid.name = "order_id";  oid.type = pq::Ty::Int64; oid.optional = true;
                     oid.numpy_type = "Int64";
    pq::Column par;  par.name = "causal_parent"; par.type = pq::Ty::Int64; par.optional = true;
                     par.numpy_type = "Int64";

    for (const LedgerRow& r : rows) {
        seq.i64.push_back(r.seq);
        trec.i64.push_back(r.t_recv_ns);
        tsnd.i64.push_back(r.has_send ? r.t_send_ns : 0);
        tsnd.present.push_back(r.has_send ? 1 : 0);
        lat.i64.push_back(r.latency_ns);
        src.i32.push_back(r.src_id);
        dst.i32.push_back(r.dst_id);
        mid.i64.push_back(r.message_id);
        mt.str.emplace_back(msg_type_name(r.msg_type));
        oid.i64.push_back(r.order_id >= 0 ? r.order_id : 0);
        oid.present.push_back(r.order_id >= 0 ? 1 : 0);
        par.i64.push_back(r.causal_parent >= 0 ? r.causal_parent : 0);
        par.present.push_back(r.causal_parent >= 0 ? 1 : 0);
    }
    cols = {seq, trec, tsnd, lat, src, dst, mid, mt, oid, par};
    pq::write(path, cols);
}

// ---------------------------------------------------------------- one scenario
struct RunOutcome {
    std::int64_t n_events = 0;
    double wall_clock_sec = 0.0;
};

static RunOutcome run_one(const std::string& scenario_path, const std::string& trace_path,
                          std::int64_t seed_override, bool have_seed_override) {
    LoadedScenario sc = load_scenario(scenario_path);
    if (have_seed_override) sc.cfg.seed = static_cast<std::uint32_t>(seed_override);

    const auto t0 = std::chrono::steady_clock::now();
    const SimResult res = run_simulation(sc.cfg, /*build_outputs=*/true);
    const std::string out_dir = dir_of(trace_path);
    write_trace_parquet(trace_path, res.trace);
    write_ledger_parquet(out_dir + "/message_trace.parquet", res.ledger);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    const std::int64_t n = static_cast<std::int64_t>(res.trace.size());
    const std::string hash = sha2::file_hex(trace_path);

    std::string j = "{\n";
    j += "  \"scenario_id\": \"" + sc.scenario_id + "\",\n";
    j += "  \"seed\": " + std::to_string(sc.cfg.seed) + ",\n";
    j += "  \"n_events\": " + std::to_string(n) + ",\n";
    j += "  \"wall_clock_sec\": " + num(wall) + ",\n";
    j += "  \"events_per_sec\": " + num(wall > 0.0 ? static_cast<double>(n) / wall : 0.0) + ",\n";
    j += "  \"trace_sha256\": \"" + hash + "\",\n";
    j += "  \"peak_memory_bytes\": " + std::to_string(peak_rss_bytes()) + ",\n";
    j += "  \"gpu_seconds\": 0.0\n";
    j += "}\n";
    if (std::FILE* f = std::fopen((out_dir + "/events.json").c_str(), "wb")) {
        std::fwrite(j.data(), 1, j.size(), f);
        std::fclose(f);
    }
    return {n, wall};
}

// ---------------------------------------------------------------- batch
static int run_batch(const std::string& batch_dir, const std::string& out_dir) {
    // Sub-scenarios are every *.json in batch_dir, in SORTED filename order -- the order
    // simulate_batch.py uses. Each is a fully independent run: run_simulation resets the global
    // order_id / message_id counters and re-seeds from the sub's own seed, which is what makes
    // the per-sub isolation gate satisfiable.
    const std::vector<std::string> subs = list_json(batch_dir);
    if (subs.empty()) {
        std::fprintf(stderr, "simulate-batch: no sub-scenarios (*.json) found in %s\n",
                     batch_dir.c_str());
        return 2;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::int64_t total = 0;
    std::string per;
    for (const std::string& sp : subs) {
        std::string stem = sp.substr(sp.find_last_of("/\\") + 1);
        if (stem.size() > 5) stem = stem.substr(0, stem.size() - 5);  // strip .json
        const std::string sub_out = out_dir + "/" + stem;
        mkdir_p(sub_out);
        const RunOutcome o = run_one(sp, sub_out + "/trace.parquet", 0, false);
        total += o.n_events;
        if (!per.empty()) per += ",\n";
        per += "    {\"sub\": \"" + stem + "\", \"n_events\": " + std::to_string(o.n_events) + "}";
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    std::string j = "{\n";
    j += "  \"n_scenarios\": " + std::to_string(subs.size()) + ",\n";
    j += "  \"total_events\": " + std::to_string(total) + ",\n";
    j += "  \"wall_clock_sec\": " + num(wall) + ",\n";
    j += "  \"events_per_sec\": " +
         num(wall > 0.0 ? static_cast<double>(total) / wall : 0.0) + ",\n";
    j += "  \"peak_memory_bytes\": " + std::to_string(peak_rss_bytes()) + ",\n";
    j += "  \"gpu_seconds\": 0.0,\n";
    j += "  \"per_scenario\": [\n" + per + "\n  ]\n";
    j += "}\n";
    if (std::FILE* f = std::fopen((out_dir + "/batch_events.json").c_str(), "wb")) {
        std::fwrite(j.data(), 1, j.size(), f);
        std::fclose(f);
    }
    std::fputs(j.c_str(), stdout);
    return 0;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    std::string verb, config, out, batch_dir, out_dir;
    std::int64_t seed = 0;
    bool have_seed = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "simulate" || a == "simulate-batch") verb = a;
        else if (a == "--config" && i + 1 < argc) config = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--batch-dir" && i + 1 < argc) batch_dir = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--seed" && i + 1 < argc) { seed = std::atoll(argv[++i]); have_seed = true; }
    }
    if (verb.empty()) verb = batch_dir.empty() ? "simulate" : "simulate-batch";

    try {
        if (verb == "simulate-batch") {
            if (batch_dir.empty() || out_dir.empty()) {
                std::fprintf(stderr,
                             "usage: simulate-batch --batch-dir DIR --out-dir DIR\n");
                return 2;
            }
            mkdir_p(out_dir);
            return run_batch(batch_dir, out_dir);
        }
        if (config.empty() || out.empty()) {
            std::fprintf(stderr, "usage: simulate --config FILE --out FILE [--seed N]\n");
            return 2;
        }
        mkdir_p(dir_of(out));
        const RunOutcome o = run_one(config, out, seed, have_seed);
        std::printf("{\"n_events\": %lld, \"wall_clock_sec\": %s}\n", (long long)o.n_events,
                    num(o.wall_clock_sec).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "simulate: %s\n", e.what());
        return 1;
    }
}
