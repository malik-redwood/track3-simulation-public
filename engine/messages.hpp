// messages.hpp — the 12 message types t3-s001 needs, plus the global id counters.
//
// message_id is a GLOBAL construction counter starting at 1, shared by every message type.
// order_id is a separate global counter starting at 0. simulate.py resets both per run:
//     Order._order_id_counter = 0
//     Message.__message_id_counter = 1
// Reproducing the construction ORDER is as important as the values, because message_id is the
// final tie-break in the kernel's delivery order.

#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <cstdint>
#include <memory>

namespace t3 {

enum class Side { Bid, Ask };

enum class MsgType {
    WakeupMsg,
    MarketHoursRequestMsg,
    MarketClosePriceRequestMsg,
    MarketHoursMsg,
    MarketClosePriceMsg,
    QuerySpreadMsg,
    QuerySpreadResponseMsg,
    LimitOrderMsg,
    CancelOrderMsg,
    OrderAcceptedMsg,
    OrderExecutedMsg,
    OrderCancelledMsg,
    MarketClosedMsg,
};

inline const char* msg_type_name(MsgType t) {
    switch (t) {
        case MsgType::WakeupMsg:                  return "AGENT_WAKEUP";
        case MsgType::MarketHoursRequestMsg:      return "MarketHoursRequestMsg";
        case MsgType::MarketClosePriceRequestMsg: return "MarketClosePriceRequestMsg";
        case MsgType::MarketHoursMsg:             return "MarketHoursMsg";
        case MsgType::MarketClosePriceMsg:        return "MarketClosePriceMsg";
        case MsgType::QuerySpreadMsg:             return "QuerySpreadMsg";
        case MsgType::QuerySpreadResponseMsg:     return "QuerySpreadResponseMsg";
        case MsgType::LimitOrderMsg:              return "LimitOrderMsg";
        case MsgType::CancelOrderMsg:             return "CancelOrderMsg";
        case MsgType::OrderAcceptedMsg:           return "OrderAcceptedMsg";
        case MsgType::OrderExecutedMsg:           return "OrderExecutedMsg";
        case MsgType::OrderCancelledMsg:          return "OrderCancelledMsg";
        case MsgType::MarketClosedMsg:            return "MarketClosedMsg";
    }
    return "?";
}

// ExchangeAgent splits inbound traffic three ways after market close: OrderMsg is refused,
// QueryMsg is explicitly allowed through, everything else is refused.
inline bool is_order_msg(MsgType t) {
    return t == MsgType::LimitOrderMsg || t == MsgType::CancelOrderMsg;
}
inline bool is_query_msg(MsgType t) { return t == MsgType::QuerySpreadMsg; }

// --------------------------------------------------------------------------- orders
struct Order {
    std::int64_t order_id = -1;
    int agent_id = 0;
    std::int64_t time_placed = 0;
    Side side = Side::Bid;
    std::int64_t quantity = 0;
    std::int64_t limit_price = 0;
    std::int64_t fill_price = 0;
    bool has_fill_price = false;
};

// Order._order_id_counter: global, post-increment, starts at 0, reset per run.
inline std::int64_t g_order_id_counter = 0;

inline std::int64_t next_order_id() { return g_order_id_counter++; }

// --------------------------------------------------------------------------- messages
struct Message {
    std::int64_t message_id = 0;
    MsgType type = MsgType::WakeupMsg;

    // The ledger records an order_id when the message carries an order; -1 means null.
    std::int64_t order_id = -1;

    Order order{};  // order-bearing messages

    // MarketHoursMsg
    std::int64_t mkt_open = 0;
    std::int64_t mkt_close = 0;

    // MarketClosePriceMsg
    std::int64_t close_price = 0;

    // QuerySpreadMsg / QuerySpreadResponseMsg
    int depth = 1;
    bool mkt_closed = false;
    std::int64_t bid = 0, bid_vol = 0, ask = 0, ask_vol = 0, last_trade = 0;
};

using MessagePtr = std::shared_ptr<Message>;

// Message.__message_id_counter: global ClassVar, starts at 1, incremented on CONSTRUCTION.
inline std::int64_t g_message_id_counter = 1;

inline void reset_abides_counters() {
    g_order_id_counter = 0;
    g_message_id_counter = 1;
}

inline MessagePtr make_message(MsgType t) {
    auto m = std::make_shared<Message>();
    m->message_id = g_message_id_counter++;
    m->type = t;
    return m;
}

// Order-bearing messages copy the order and expose its id to the ledger.
inline MessagePtr make_order_message(MsgType t, const Order& o) {
    auto m = make_message(t);
    m->order = o;
    m->order_id = o.order_id;
    return m;
}

}  // namespace t3

#endif  // MESSAGES_HPP
