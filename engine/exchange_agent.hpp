// exchange_agent.hpp — ExchangeAgent, bootstrap + spread queries.
//
// The order book is NOT here yet; get_l1 returns an empty book. That is enough to validate the
// startup handshake (the first 21 ledger rows of t3-s001) before the matching engine exists.
//
// TWO DELAY FACTS, both verified against the t3-s001 ledger:
//
//  * ExchangeAgent.receive_message calls set_computation_delay(self.computation_delay) on EVERY
//    inbound message, and the adapter passes computation_delay = 0 unless
//    exchange_config.protocol_enforcement is set. So the exchange replies with sent_time ==
//    current_time, while traders carry the kernel default of 50 ns.
//    Confirmed: agent 1's request is sent at date_ns+50 and delivered at +176 (latency 126); the
//    exchange's reply is sent at exactly +176, so its own delay was 0.
//
//  * pipeline_delay is added ONLY to OrderAcceptedMsg / OrderCancelledMsg / OrderExecutedMsg,
//    and the adapter passes 0 for it too (non-protocol mode).
//
//  * The exchange's mkt_close wakeup is created in kernel_initializing, which is why it holds
//    message_id 1 -- ahead of every start-time wakeup.

#ifndef EXCHANGE_AGENT_HPP
#define EXCHANGE_AGENT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "agent.hpp"
#include "messages.hpp"
#include "order_book.hpp"

namespace t3 {

class ExchangeAgent : public Agent, public BookOwner {
public:
    ExchangeAgent(int id, nprng::LegacyRandomState rs, std::int64_t mkt_open,
                  std::int64_t mkt_close, std::int64_t open_price, int computation_delay,
                  int pipeline_delay, std::string stp_policy = "")
        : Agent(id, rs), mkt_open_(mkt_open), mkt_close_(mkt_close),
          computation_delay_(computation_delay), pipeline_delay_(pipeline_delay),
          stp_policy_(std::move(stp_policy)), book_(this, open_price) {}

    // --- BookOwner -------------------------------------------------------
    // pipeline_delay applies ONLY to OrderAccepted / OrderCancelled / OrderExecuted, and the
    // adapter passes 0 for it in non-protocol mode.
    void book_send(int recipient_id, const MessagePtr& m) override {
        const bool order_report = m->type == MsgType::OrderAcceptedMsg ||
                                  m->type == MsgType::OrderCancelledMsg ||
                                  m->type == MsgType::OrderExecutedMsg;
        kernel_->send_message(id_, recipient_id, m, order_report ? pipeline_delay_ : 0);
    }

    void book_log_quote(const char* event_type, std::int64_t price,
                        std::int64_t volume) override {
        LogEvent e;
        e.event_time = current_time_;
        e.agent_id = id_;
        e.event_type = event_type;  // "BEST_BID" / "BEST_ASK"
        e.quote_price = price;
        e.quote_volume = volume;
        e.order_id = -1;
        log_.push_back(e);
    }

    std::int64_t book_now() const override { return current_time_; }

    const std::string& book_stp_policy() const override { return stp_policy_; }

    // ExchangeAgent.kernel_initializing ends with set_wakeup(self.mkt_close).
    void kernel_initializing() override {
        kernel_->set_wakeup(id_, mkt_close_);
    }

    void wakeup(std::int64_t current_time) override {
        Agent::wakeup(current_time);
        if (current_time >= mkt_close_ && !close_sent_) {
            close_sent_ = true;
            // ONE MarketClosePriceMsg object delivered to every subscriber: one message_id,
            // N send_message calls, N latency draws. Verified: 4 rows share 1 id in t3-s001.
            auto m = make_message(MsgType::MarketClosePriceMsg);
            m->close_price = book_.last_trade();
            for (int sub : close_price_subscribers_) {
                kernel_->send_message(id_, sub, m);
            }
        }
    }

    void receive_message(std::int64_t current_time, int sender_id,
                         const MessagePtr& m) override {
        Agent::receive_message(current_time, sender_id, m);
        // MUST precede any send: the reply's sent_time uses this value.
        kernel_->set_computation_delay(id_, computation_delay_);

        // Post-close gate (exchange_agent.py:331). STRICTLY greater than mkt_close.
        // OrderMsg is refused outright -- no book processing, so no acceptance, no quote log
        // and no fills. QueryMsg is deliberately still served so agents can read the closing
        // trade. Everything else is refused too. Omitting this produced 4 phantom events on
        // t3-eq001-pareto-latency-tail, whose 80 ms latency tail lands orders after the close.
        if (current_time > mkt_close_) {
            if (is_order_msg(m->type) || !is_query_msg(m->type)) {
                kernel_->send_message(id_, sender_id, make_message(MsgType::MarketClosedMsg));
                return;
            }
        }

        switch (m->type) {
            case MsgType::MarketHoursRequestMsg: {
                auto r = make_message(MsgType::MarketHoursMsg);
                r->mkt_open = mkt_open_;
                r->mkt_close = mkt_close_;
                kernel_->send_message(id_, sender_id, r);
                break;
            }
            case MsgType::MarketClosePriceRequestMsg:
                close_price_subscribers_.push_back(sender_id);
                break;
            case MsgType::QuerySpreadMsg: {
                auto r = make_message(MsgType::QuerySpreadResponseMsg);
                r->depth = m->depth;
                r->mkt_closed = current_time > mkt_close_;
                r->bid = book_.best_bid();
                r->bid_vol = book_.best_bid_vol();
                r->ask = book_.best_ask();
                r->ask_vol = book_.best_ask_vol();
                r->last_trade = book_.last_trade();
                kernel_->send_message(id_, sender_id, r);
                break;
            }
            case MsgType::LimitOrderMsg:
                book_.handle_limit_order(m->order);
                break;
            case MsgType::CancelOrderMsg:
                book_.cancel_order(m->order);
                break;
            default:
                break;
        }
    }

protected:
    std::int64_t mkt_open_, mkt_close_;
    int computation_delay_, pipeline_delay_;
    bool close_sent_ = false;
    std::string stp_policy_;
    std::vector<int> close_price_subscribers_;
    OrderBook book_;
};

// --------------------------------------------------------------------------- agents.py
// ScheduledAgent: wake every interval_ns, re-arm, query the spread, act on the response.
class ScheduledAgent : public TradingAgent {
public:
    ScheduledAgent(int id, nprng::LegacyRandomState rs, int exchange_id, std::int64_t interval_ns)
        : TradingAgent(id, rs, exchange_id, /*log_orders=*/true),
          interval_ns_(interval_ns < 1 ? 1 : interval_ns) {}

    void wakeup(std::int64_t current_time) override {
        TradingAgent::wakeup(current_time);
        // `if not self.mkt_open or not self.mkt_close or self.mkt_closed: return`
        if (!mkt_open_known_ || mkt_closed_) return;
        kernel_->set_wakeup(id_, current_time + interval_ns_);
        get_current_spread(1);
        awaiting_spread_ = true;
    }

    void on_spread_response() override {
        if (awaiting_spread_) {
            if (!mkt_closed_) act();
            awaiting_spread_ = false;
        }
    }

    virtual void act() = 0;

protected:
    std::int64_t interval_ns_;
    bool awaiting_spread_ = false;
};

}  // namespace t3

#endif  // EXCHANGE_AGENT_HPP
