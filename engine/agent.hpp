// agent.hpp — Agent / TradingAgent base, mirroring abides-core agent.py and
// abides-markets agents/trading_agent.py at pinned commit f9cbe51.
//
// The bootstrap handshake reproduced here is verified against the t3-s001 ledger:
//
//   kernel_initializing (ALL agents, in id order)
//     ExchangeAgent: set_wakeup(mkt_close)                 -> message_id 1
//   kernel_starting (ALL agents, in id order)
//     Agent::kernel_starting: set_wakeup(start_time)       -> ids 2..1+n_agents
//   first wakeup, per trader, in THIS order:
//     MarketClosePriceRequestMsg  (the `if first_wake` block)   -> odd id of the pair
//     MarketHoursRequestMsg       (the `if mkt_open is None` block)
//   on MarketHoursMsg:
//     set_wakeup(mkt_open + get_wake_frequency()), which is 0 for ScheduledAgent
//
// start_time is MIDNIGHT (date_ns), not mkt_open, so the first wakeup lands long before the
// market opens and the agent cannot act on it.

#ifndef AGENT_HPP
#define AGENT_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kernel.hpp"
#include "messages.hpp"
#include "numpy_legacy_rng.hpp"

namespace t3 {

// One row of an agent's event log; trace.py turns these into trace.parquet rows.
struct LogEvent {
    std::int64_t event_time = 0;
    int agent_id = 0;
    const char* event_type = "";
    // order-lifecycle payload
    std::int64_t order_id = -1;
    Side side = Side::Bid;
    std::int64_t quantity = 0;
    std::int64_t limit_price = 0;
    std::int64_t fill_price = 0;
    bool has_fill_price = false;
    // BEST_BID / BEST_ASK payload ("SYM,price,volume")
    std::int64_t quote_price = 0;
    std::int64_t quote_volume = 0;
};

class Agent {
public:
    Agent(int id, nprng::LegacyRandomState rs) : id_(id), random_state_(rs) {}
    virtual ~Agent() = default;

    int id() const { return id_; }
    const std::vector<LogEvent>& log() const { return log_; }

    void attach(Kernel* k) { kernel_ = k; }

    virtual void kernel_initializing() {}

    // Agent.kernel_starting: base schedules a wakeup for the first available timestamp.
    virtual void kernel_starting(std::int64_t start_time) {
        kernel_->set_wakeup(id_, start_time);
    }

    virtual void kernel_stopping() {}

    virtual void wakeup(std::int64_t current_time) { current_time_ = current_time; }
    virtual void receive_message(std::int64_t current_time, int /*sender_id*/,
                                 const MessagePtr& /*m*/) {
        current_time_ = current_time;
    }

protected:
    void log_event(const char* type) {
        LogEvent e;
        e.event_time = current_time_;
        e.agent_id = id_;
        e.event_type = type;
        log_.push_back(e);
    }
    void log_order_event(const char* type, const Order& o, bool as_fill) {
        LogEvent e;
        e.event_time = current_time_;
        e.agent_id = id_;
        e.event_type = type;
        e.order_id = o.order_id;
        e.side = o.side;
        e.quantity = o.quantity;
        e.limit_price = o.limit_price;
        e.fill_price = o.fill_price;
        e.has_fill_price = as_fill;
        log_.push_back(e);
    }

    int id_;
    Kernel* kernel_ = nullptr;
    nprng::LegacyRandomState random_state_;
    std::int64_t current_time_ = 0;
    std::vector<LogEvent> log_;
};

// --------------------------------------------------------------------------- TradingAgent
class TradingAgent : public Agent {
public:
    TradingAgent(int id, nprng::LegacyRandomState rs, int exchange_id, bool log_orders)
        : Agent(id, rs), exchange_id_(exchange_id), log_orders_(log_orders) {}

    void wakeup(std::int64_t current_time) override {
        Agent::wakeup(current_time);
        if (first_wake_) {
            first_wake_ = false;
            // `if self.first_wake:` runs BEFORE the market-hours request -- confirmed by the
            // ledger id pairs (7,8) (9,10) (11,12) (13,14).
            kernel_->send_message(id_, exchange_id_,
                                  make_message(MsgType::MarketClosePriceRequestMsg));
        }
        if (!mkt_open_known_) {
            kernel_->send_message(id_, exchange_id_,
                                  make_message(MsgType::MarketHoursRequestMsg));
        }
    }

    void receive_message(std::int64_t current_time, int sender_id,
                         const MessagePtr& m) override {
        Agent::receive_message(current_time, sender_id, m);
        switch (m->type) {
            case MsgType::MarketHoursMsg: {
                const bool was_unknown = !mkt_open_known_;
                mkt_open_ = m->mkt_open;
                mkt_close_ = m->mkt_close;
                mkt_open_known_ = true;
                if (was_unknown) {
                    kernel_->set_wakeup(id_, mkt_open_ + get_wake_frequency());
                }
                break;
            }
            case MsgType::MarketClosePriceMsg:
                last_trade_ = m->close_price;
                break;
            case MsgType::MarketClosedMsg:
                // TradingAgent.market_closed(): sets the flag and logs MKT_CLOSED, which is not
                // one of the trace msg_types so it produces no trace row.
                mkt_closed_ = true;
                break;
            case MsgType::OrderAcceptedMsg:
                if (log_orders_) log_order_event("ORDER_ACCEPTED", m->order, false);
                break;
            case MsgType::OrderExecutedMsg:
                on_order_executed(m->order);
                break;
            case MsgType::OrderCancelledMsg:
                on_order_cancelled(m->order);
                break;
            case MsgType::QuerySpreadResponseMsg:
                known_bid_ = m->bid;
                known_bid_vol_ = m->bid_vol;
                known_ask_ = m->ask;
                known_ask_vol_ = m->ask_vol;
                last_trade_ = m->last_trade;
                if (m->mkt_closed) mkt_closed_ = true;
                on_spread_response();
                break;
            default:
                break;
        }
    }

    virtual std::int64_t get_wake_frequency() const { return 0; }
    virtual void on_spread_response() {}

protected:
    // TradingAgent.place_limit_order: construct the Order (burning an order id), store a COPY in
    // self.orders, then send. Order construction precedes message construction.
    void place_limit_order(std::int64_t quantity, Side side, std::int64_t limit_price) {
        if (quantity <= 0) return;
        Order o;
        o.order_id = next_order_id();
        o.agent_id = id_;
        o.time_placed = current_time_;
        o.side = side;
        o.quantity = quantity;
        o.limit_price = limit_price;
        orders_[o.order_id] = o;
        order_insertion_.push_back(o.order_id);
        if (log_orders_) log_order_event("ORDER_SUBMITTED", o, false);
        kernel_->send_message(id_, exchange_id_, make_order_message(MsgType::LimitOrderMsg, o));
    }

    // cancel_all_orders iterates self.orders.values() -- a dict, so INSERTION order, not id
    // order. The entry is not removed until OrderCancelledMsg comes back.
    void cancel_all_orders() {
        for (std::int64_t oid : order_insertion_) {
            auto it = orders_.find(oid);
            if (it == orders_.end()) continue;
            kernel_->send_message(id_, exchange_id_,
                                  make_order_message(MsgType::CancelOrderMsg, it->second));
        }
    }

    void get_current_spread(int depth = 1) {
        auto m = make_message(MsgType::QuerySpreadMsg);
        m->depth = depth;
        kernel_->send_message(id_, exchange_id_, m);
    }

    void on_order_executed(const Order& o) {
        if (log_orders_) log_order_event("ORDER_EXECUTED", o, true);
        auto it = orders_.find(o.order_id);
        if (it != orders_.end()) {
            if (it->second.quantity <= o.quantity) {
                erase_order(o.order_id);
            } else {
                it->second.quantity -= o.quantity;
            }
        }
    }

    void on_order_cancelled(const Order& o) {
        if (log_orders_) log_order_event("ORDER_CANCELLED", o, false);
        erase_order(o.order_id);
    }

    void erase_order(std::int64_t oid) {
        orders_.erase(oid);
        for (std::size_t i = 0; i < order_insertion_.size(); ++i) {
            if (order_insertion_[i] == oid) {
                order_insertion_.erase(order_insertion_.begin() + static_cast<long>(i));
                break;
            }
        }
    }

    int exchange_id_;
    bool log_orders_;
    bool first_wake_ = true;
    bool mkt_open_known_ = false;
    bool mkt_closed_ = false;
    std::int64_t mkt_open_ = 0, mkt_close_ = 0;
    std::int64_t known_bid_ = 0, known_bid_vol_ = 0, known_ask_ = 0, known_ask_vol_ = 0;
    std::int64_t last_trade_ = 0;

    std::map<std::int64_t, Order> orders_;       // value lookup
    std::vector<std::int64_t> order_insertion_;  // dict insertion order
};

}  // namespace t3

#endif  // AGENT_HPP
