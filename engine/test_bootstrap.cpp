// test_bootstrap.cpp — reproduce the t3-s001 startup handshake and dump the ledger.
//
// The order book does not exist yet, so only the bootstrap can match: ledger rows 0..20 of
// t3-s001, covering the start-time wakeups, the market-hours handshake, and the mkt_open
// wakeups. Row 21 onwards is QuerySpreadMsg traffic that depends on a real book.
//
// Emits one line per ledger row:
//     <seq> <message_id> <src> <dst> <t_send|-> <t_recv> <latency> <msg_type>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "agent.hpp"
#include "exchange_agent.hpp"
#include "kernel.hpp"
#include "messages.hpp"
#include "numpy_legacy_rng.hpp"
#include "oracle.hpp"
#include "trace.hpp"

using namespace t3;

// NoiseTrader / MarketMaker, per abides_fork/agents.py. Order placement is stubbed out for the
// bootstrap test -- the draws still happen so the streams stay aligned, but with no book the
// prices would be meaningless anyway.
class NoiseTrader : public ScheduledAgent {
public:
    NoiseTrader(int id, nprng::LegacyRandomState rs, int ex, std::int64_t interval,
                double size_mean, double size_std, int offset_ticks, std::int64_t ref_price)
        : ScheduledAgent(id, rs, ex, interval), size_mean_(size_mean), size_std_(size_std),
          offset_ticks_(offset_ticks), ref_price_(ref_price) {}

    void act() override {
        const std::int64_t size =
            std::max<std::int64_t>(1, nprng::py_round_to_i64(random_state_.normal(size_mean_, size_std_)));
        const bool buy = random_state_.randint(0, 2) != 0;
        const std::int64_t off = random_state_.randint(0, offset_ticks_ + 1);
        if (buy) {
            const std::int64_t anchor = known_ask_ ? known_ask_ : (known_bid_ ? known_bid_ : ref_price_);
            place_limit_order(size, Side::Bid, anchor + off);
        } else {
            const std::int64_t anchor = known_bid_ ? known_bid_ : (known_ask_ ? known_ask_ : ref_price_);
            place_limit_order(size, Side::Ask, anchor - off);
        }
    }

private:
    double size_mean_, size_std_;
    int offset_ticks_;
    std::int64_t ref_price_;
};

class MarketMaker : public ScheduledAgent {
public:
    MarketMaker(int id, nprng::LegacyRandomState rs, int ex, std::int64_t interval,
                int spread_ticks, int depth_levels, std::int64_t size_per_level,
                std::int64_t ref_price)
        : ScheduledAgent(id, rs, ex, interval),
          spread_ticks_(spread_ticks < 2 ? 2 : spread_ticks),
          depth_levels_(depth_levels < 1 ? 1 : depth_levels),
          size_per_level_(size_per_level < 1 ? 1 : size_per_level), ref_price_(ref_price) {}

    void act() override {
        const std::int64_t mid = (known_bid_ && known_ask_) ? ((known_bid_ + known_ask_) / 2)
                                                            : ref_price_;
        cancel_all_orders();
        const std::int64_t half = spread_ticks_ / 2;
        for (int lvl = 0; lvl < depth_levels_; ++lvl) {
            place_limit_order(size_per_level_, Side::Bid, mid - half - lvl);
            place_limit_order(size_per_level_, Side::Ask, mid + half + lvl);
        }
    }

private:
    int spread_ticks_, depth_levels_;
    std::int64_t size_per_level_, ref_price_;
};

int main(int argc, char** argv) {
    const bool emit_trace = (argc > 1 && std::string(argv[1]) == "--trace");
    reset_abides_counters();

    // --- t3-s001-price-time-priority scenario constants -----------------------
    const std::uint32_t scenario_seed = 1001;
    const std::int64_t date_ns = 1612483200000000000LL;
    const std::int64_t mkt_open = date_ns + 34200000000000LL;  // 09:30:00
    const std::int64_t horizon_ns = 10000000000LL;             // 10 s
    const std::int64_t mkt_close = mkt_open + horizon_ns;
    const std::int64_t stop_time = mkt_close + 1000000000LL;   // + 1s
    const std::int64_t ref_price = 100000;

    // --- the verified global draw chain --------------------------------------
    nprng::LegacyRandomState global(scenario_seed);
    nprng::LegacyRandomState oracle_rs(global.draw_seed());   // word 1
    SymbolParams sp;
    sp.r_bar = static_cast<double>(ref_price);
    sp.kappa = 0.05 / 1e9;
    sp.fund_vol = 0.0005;
    sp.megashock_lambda_a = 2.77778e-18;  // jump_intensity == 0 -> rmsc04 fallback
    sp.megashock_mean = 1000.0;
    sp.megashock_var = 50000.0;
    SparseMeanRevertingOracle oracle(mkt_open, date_ns + 57600000000000LL /*16:00*/, sp,
                                     &oracle_rs, &global);   // consumes words 2,3
    nprng::LegacyRandomState exchange_rs(global.draw_seed());  // word 4

    std::vector<nprng::LegacyRandomState> agent_rs;
    for (int i = 0; i < 4; ++i) agent_rs.emplace_back(global.draw_seed());  // words 5..8
    const std::uint32_t latency_seed = global.draw_seed();                   // word 9
    const std::uint32_t kernel_seed = global.draw_seed();                    // word 10
    (void)kernel_seed;  // unused when a latency_config is present

    // --- kernel --------------------------------------------------------------
    Kernel kernel(date_ns, stop_time, /*default_computation_delay=*/50);
    kernel.set_latency_model(LatencyModel(LatencyModel::Kind::Uniform, /*mean*/ 500.0,
                                          /*sigma*/ 0.0, /*min*/ 100.0, /*max*/ 2000.0,
                                          /*alpha*/ 1.5, latency_seed));

    ExchangeAgent exchange(0, exchange_rs, mkt_open, mkt_close, oracle.get_daily_open_price(),
                           /*computation_delay=*/0, /*pipeline_delay=*/0);
    std::vector<std::unique_ptr<ScheduledAgent>> traders;
    for (int i = 0; i < 3; ++i) {
        traders.push_back(std::make_unique<NoiseTrader>(
            1 + i, agent_rs[static_cast<std::size_t>(i)], 0,
            /*interval*/ static_cast<std::int64_t>(1e9 / 2.0), 10.0, 2.0, 5, ref_price));
    }
    traders.push_back(std::make_unique<MarketMaker>(4, agent_rs[3], 0,
                                                    /*rebalance*/ 1000000000LL, 2, 3, 10,
                                                    ref_price));

    kernel.add_agent(&exchange);
    for (auto& t : traders) kernel.add_agent(t.get());

    kernel.run();

    if (emit_trace) {
        // parse_logs_df order: agents in ID order, each agent's log chronological.
        std::vector<LogEvent> events;
        for (const LogEvent& e : exchange.log()) events.push_back(e);
        for (auto& t : traders)
            for (const LogEvent& e : t->log()) events.push_back(e);
        for (const TraceRow& r : extract_trace(events)) {
            std::printf("%lld %d %s %s %lld %lld %lld\n", (long long)r.t_ns, r.agent_id,
                        r.msg_type, r.side[0] ? r.side : "-", (long long)r.price,
                        (long long)r.size, (long long)r.order_id);
        }
        return 0;
    }

    // All 10 message_trace columns; "-" encodes a null.
    for (const LedgerRow& r : kernel.delivered_ledger_sorted()) {
        char t_send[32], oid[32], parent[32];
        if (r.has_send) std::snprintf(t_send, sizeof t_send, "%lld", (long long)r.t_send_ns);
        else std::snprintf(t_send, sizeof t_send, "-");
        if (r.order_id >= 0) std::snprintf(oid, sizeof oid, "%lld", (long long)r.order_id);
        else std::snprintf(oid, sizeof oid, "-");
        if (r.causal_parent >= 0)
            std::snprintf(parent, sizeof parent, "%lld", (long long)r.causal_parent);
        else std::snprintf(parent, sizeof parent, "-");
        std::printf("%lld %lld %s %lld %d %d %lld %s %s %s\n", (long long)r.seq,
                    (long long)r.t_recv_ns, t_send, (long long)r.latency_ns, r.src_id, r.dst_id,
                    (long long)r.message_id, msg_type_name(r.msg_type), oid, parent);
    }
    return 0;
}
