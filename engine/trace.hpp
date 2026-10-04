// trace.hpp — build the canonical 7-column trace from the agent event logs.
//
// Mirrors abides_fork/trace.py::extract_trace.
//
// ROW ORDER IS DERIVED, NOT CHOSEN, AND IT MATTERS FOR TIER A:
//
//  * parse_logs_df concatenates agents in ID ORDER, each agent's log chronological. That is the
//    "file order" every subsequent STABLE sort falls back on for ties.
//
//  * Order events are stable-sorted by EventTime, then the LAST ORDER_EXECUTED per order_id (in
//    that sorted order) becomes ORDER_FILLED and the earlier ones PARTIAL_FILL. This is
//    positional, not quantity based -- an order partially filled and then cancelled still ends
//    in ORDER_FILLED.
//
//  * QUOTE_UPDATE rows come from BEST_BID / BEST_ASK, deduplicated per (t_ns, side) keeping the
//    LAST value, but ordered by each key's FIRST appearance.
//
//  * Final sort is (t_ns, order_id), stable. Quotes carry order_id = -1, so every QUOTE_UPDATE
//    sorts BEFORE every order event at the same nanosecond.
//
//  * Emitting two quote rows for one (t_ns, side) is instantly inadmissible: mid_price_series
//    indexes on t_ns and raises ValueError, which the official gate converts to T3_PARSE_ERROR.

#ifndef TRACE_HPP
#define TRACE_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "agent.hpp"

namespace t3 {

struct TraceRow {
    std::int64_t t_ns = 0;
    int agent_id = 0;
    const char* msg_type = "";
    const char* side = "";  // "BID" / "ASK" / "" for null
    std::int64_t price = 0;
    std::int64_t size = 0;
    std::int64_t order_id = -1;
};

inline const char* side_str(Side s) { return s == Side::Bid ? "BID" : "ASK"; }

// `events` must already be the parse_logs_df concatenation: agents in id order, each agent's log
// in chronological order.
inline std::vector<TraceRow> extract_trace(const std::vector<LogEvent>& events) {
    // ---- order-lifecycle events -----------------------------------------
    struct Pending {
        const LogEvent* e;
        std::size_t file_pos;  // stable-sort fallback
        bool is_exec;
    };
    std::vector<Pending> orders;
    orders.reserve(events.size());
    for (std::size_t i = 0; i < events.size(); ++i) {
        const LogEvent& e = events[i];
        const bool is_exec = std::strcmp(e.event_type, "ORDER_EXECUTED") == 0;
        const bool keep = is_exec || std::strcmp(e.event_type, "ORDER_SUBMITTED") == 0 ||
                          std::strcmp(e.event_type, "ORDER_ACCEPTED") == 0 ||
                          std::strcmp(e.event_type, "ORDER_CANCELLED") == 0 ||
                          std::strcmp(e.event_type, "ORDER_REPLACED") == 0;
        if (keep && e.order_id >= 0) orders.push_back({&e, i, is_exec});
    }
    std::stable_sort(orders.begin(), orders.end(), [](const Pending& a, const Pending& b) {
        return a.e->event_time < b.e->event_time;
    });

    // The last ORDER_EXECUTED per order_id, in the sorted order, is the ORDER_FILLED.
    std::map<std::int64_t, std::size_t> last_exec_pos;
    for (std::size_t i = 0; i < orders.size(); ++i) {
        if (orders[i].is_exec) last_exec_pos[orders[i].e->order_id] = i;
    }

    std::vector<TraceRow> out;
    out.reserve(orders.size() + 64);
    for (std::size_t i = 0; i < orders.size(); ++i) {
        const LogEvent& e = *orders[i].e;
        TraceRow r;
        r.t_ns = e.event_time;
        r.agent_id = e.agent_id;
        r.side = side_str(e.side);
        r.size = e.quantity;
        r.order_id = e.order_id;
        if (orders[i].is_exec) {
            const bool final_exec = last_exec_pos[e.order_id] == i;
            r.msg_type = final_exec ? "ORDER_FILLED" : "PARTIAL_FILL";
            r.price = e.fill_price;  // executions carry the fill price
        } else {
            r.msg_type = e.event_type;
            r.price = e.limit_price;
        }
        out.push_back(r);
    }

    // ---- quote events ----------------------------------------------------
    // Built from the RAW (unsorted) event order, which for quotes is the exchange's own
    // chronological log. keep=last per (t_ns, side), ordered by the key's first appearance.
    std::vector<TraceRow> quotes;
    std::map<std::pair<std::int64_t, int>, std::size_t> slot;  // (t_ns, side) -> index in quotes
    std::vector<std::pair<std::int64_t, int>> first_seen;
    for (const LogEvent& e : events) {
        const bool is_bid = std::strcmp(e.event_type, "BEST_BID") == 0;
        const bool is_ask = std::strcmp(e.event_type, "BEST_ASK") == 0;
        if (!is_bid && !is_ask) continue;
        TraceRow r;
        r.t_ns = e.event_time;
        r.agent_id = e.agent_id;
        r.msg_type = "QUOTE_UPDATE";
        r.side = is_bid ? "BID" : "ASK";
        r.price = e.quote_price;
        r.size = e.quote_volume;
        r.order_id = -1;
        const std::pair<std::int64_t, int> key{e.event_time, is_bid ? 0 : 1};
        auto it = slot.find(key);
        if (it == slot.end()) {
            slot[key] = quotes.size();
            quotes.push_back(r);          // first appearance fixes the position
        } else {
            quotes[it->second] = r;       // keep=last fixes the value
        }
    }
    out.insert(out.end(), quotes.begin(), quotes.end());

    // ---- final canonical sort -------------------------------------------
    // (t_ns, order_id), STABLE, so order events keep their relative order and quotes (order_id
    // -1) land ahead of order events sharing a timestamp.
    std::stable_sort(out.begin(), out.end(), [](const TraceRow& a, const TraceRow& b) {
        if (a.t_ns != b.t_ns) return a.t_ns < b.t_ns;
        return a.order_id < b.order_id;
    });
    return out;
}

}  // namespace t3

#endif  // TRACE_HPP
