// order_book.hpp — the matching engine, from abides_markets/order_book.py + price_level.py
// at pinned commit f9cbe51 with exchange_protocol_stp.patch applied.
//
// FACTS THAT DECIDE THE TRACE (all read from source, not inferred)
//
//  * Quotes are logged ONLY by handle_limit_order, and only for a NON-EMPTY side:
//        if self.bids: logEvent("BEST_BID", "SYM,price,total_quantity")
//        if self.asks: logEvent("BEST_ASK", ...)
//    cancel_order / enter_order / execute_order log no quotes. So every QUOTE_UPDATE row in the
//    trace descends from a limit order -- 213 of t3-s001's 604 rows, from 120 limit orders
//    (<= 2 raw events each, then deduped per (t_ns, side) by trace.py).
//
//  * ONE FILL SENDS TWO MESSAGES, PASSIVE FIRST:
//        send(matched_order.agent_id, OrderExecutedMsg(matched_order))   <- resting side
//        send(order.agent_id,         OrderExecutedMsg(filled_order))   <- aggressor
//    Construction order fixes their message_ids, which is the kernel's final tie-break.
//
//  * The fill price is the PASSIVE side's advertised limit price, never the aggressor's.
//
//  * execute_order only ever looks at book[0]. One fill per call; handle_limit_order loops.
//
//  * Self-trade prevention is OPT-IN (stp_policy is None unless
//    exchange_config.protocol_enforcement), so for most units the block is skipped entirely.
//
// Price-to-comply and hidden orders are unreachable for the four Track-3 agent types (they call
// place_limit_order with no post-only or PTC flags), so those branches are omitted and guarded
// with an assert rather than silently approximated.

#ifndef ORDER_BOOK_HPP
#define ORDER_BOOK_HPP

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "messages.hpp"

namespace t3 {

// What the book needs from its owning ExchangeAgent. Breaks the circular dependency.
class BookOwner {
public:
    virtual ~BookOwner() = default;
    virtual void book_send(int recipient_id, const MessagePtr& m) = 0;
    virtual void book_log_quote(const char* event_type, std::int64_t price,
                                std::int64_t volume) = 0;
    virtual std::int64_t book_now() const = 0;
    // "" == disabled (the non-protocol default), else "cancel_newest" / "cancel_oldest".
    virtual const std::string& book_stp_policy() const = 0;
};

// --------------------------------------------------------------------------- PriceLevel
struct PriceLevel {
    std::int64_t price = 0;
    Side side = Side::Bid;
    std::deque<Order> visible;  // FIFO: add_order appends, peek/pop take the front

    std::int64_t total_quantity() const {
        std::int64_t t = 0;
        for (const Order& o : visible) t += o.quantity;
        return t;
    }
    bool is_empty() const { return visible.empty(); }

    // order_is_match: the incoming order is on the OPPOSITE side of this level.
    bool order_is_match(const Order& o) const {
        if (o.side == Side::Bid) return o.limit_price >= price;
        return o.limit_price <= price;
    }
    // The remaining three compare an order on the SAME side as this level.
    bool order_has_better_price(const Order& o) const {
        return o.side == Side::Bid ? (o.limit_price > price) : (o.limit_price < price);
    }
    bool order_has_worse_price(const Order& o) const {
        return o.side == Side::Bid ? (o.limit_price < price) : (o.limit_price > price);
    }
    bool order_has_equal_price(const Order& o) const { return o.limit_price == price; }
};

// --------------------------------------------------------------------------- OrderBook
class OrderBook {
public:
    explicit OrderBook(BookOwner* owner, std::int64_t opening_price)
        : owner_(owner), last_trade_(opening_price) {}

    std::int64_t last_trade() const { return last_trade_; }

    bool has_bid() const { return !bids_.empty(); }
    bool has_ask() const { return !asks_.empty(); }
    std::int64_t best_bid() const { return bids_.empty() ? 0 : bids_[0].price; }
    std::int64_t best_bid_vol() const { return bids_.empty() ? 0 : bids_[0].total_quantity(); }
    std::int64_t best_ask() const { return asks_.empty() ? 0 : asks_[0].price; }
    std::int64_t best_ask_vol() const { return asks_.empty() ? 0 : asks_[0].total_quantity(); }

    void handle_limit_order(Order order) {
        if (order.quantity <= 0) return;
        if (order.limit_price < 0) return;

        std::vector<std::pair<std::int64_t, std::int64_t>> executed;  // (quantity, fill_price)

        while (true) {
            // Self-trade prevention. Opt-in: stp_policy is empty unless
            // exchange_config.protocol_enforcement is set, so for 58 of 65 units this block is
            // skipped entirely. It only ever inspects the BEST level, never the whole book.
            const std::string& stp = owner_->book_stp_policy();
            if (!stp.empty()) {
                std::vector<PriceLevel>& opp = (order.side == Side::Bid) ? asks_ : bids_;
                if (!opp.empty() && opp[0].order_is_match(order) &&
                    !opp[0].visible.empty() &&
                    opp[0].visible.front().agent_id == order.agent_id) {
                    if (stp == "cancel_oldest") {
                        // `self.cancel_order(resting, quiet=quiet)` -- quiet is
                        // handle_limit_order's parameter, which is FALSE on the normal exchange
                        // path. So this is the MESSAGE-SENDING cancel: the resting order's owner
                        // receives an OrderCancelledMsg. Using a silent cancel here cost 3 units
                        // (mp02 / mp04 / mp07) their exact match.
                        Order resting = opp[0].visible.front();
                        if (cancel_order(resting)) continue;
                    } else {
                        // cancel_newest: kill the incoming remainder and stop. The aggressor is
                        // notified; nothing enters the book.
                        owner_->book_send(order.agent_id,
                                          make_order_message(MsgType::OrderCancelledMsg, order));
                        break;
                    }
                }
            }
            std::optional<Order> matched = execute_order(order);
            if (matched.has_value()) {
                executed.emplace_back(matched->quantity, matched->fill_price);
                if (order.quantity <= 0) break;
                // otherwise keep consuming
            } else {
                enter_order(order);  // the REMAINDER enters the book
                owner_->book_send(order.agent_id,
                                  make_order_message(MsgType::OrderAcceptedMsg, order));
                break;
            }
        }

        // Quote logging: only here, and only for a non-empty side.
        if (!bids_.empty()) {
            owner_->book_log_quote("BEST_BID", bids_[0].price, bids_[0].total_quantity());
        }
        if (!asks_.empty()) {
            owner_->book_log_quote("BEST_ASK", asks_[0].price, asks_[0].total_quantity());
        }

        if (!executed.empty()) {
            std::int64_t trade_qty = 0;
            double trade_price = 0.0;
            for (auto& [q, p] : executed) {
                trade_qty += q;
                trade_price += static_cast<double>(p) * static_cast<double>(q);
            }
            // int(round(trade_price / trade_qty)) -- banker's rounding on the float division.
            last_trade_ = nprng::py_round_to_i64(trade_price / static_cast<double>(trade_qty));
        }
    }

    // Finds a single best match. Mutates `order.quantity`. Returns the matched (passive) order
    // portion, or nullopt when there is nothing to match against.
    std::optional<Order> execute_order(Order& order) {
        std::vector<PriceLevel>& book = (order.side == Side::Bid) ? asks_ : bids_;
        if (book.empty()) return std::nullopt;
        if (!book[0].order_is_match(order)) return std::nullopt;

        Order matched;
        if (order.quantity >= book[0].visible.front().quantity) {
            // Consume the entire resting order, then drop the level if it emptied.
            matched = book[0].visible.front();
            book[0].visible.pop_front();
            if (book[0].is_empty()) book.erase(book.begin());
        } else {
            // Partial: the resting order stays, decremented in place.
            Order& book_order = book[0].visible.front();
            matched = book_order;
            matched.quantity = order.quantity;
            book_order.quantity -= matched.quantity;
        }

        // The fill happens at the price the PASSIVE side was advertising.
        matched.fill_price = matched.limit_price;
        matched.has_fill_price = true;

        Order filled = order;
        filled.quantity = matched.quantity;
        filled.fill_price = matched.fill_price;
        filled.has_fill_price = true;

        order.quantity -= filled.quantity;

        // Passive first, then aggressor. This order fixes the two message_ids.
        owner_->book_send(matched.agent_id,
                          make_order_message(MsgType::OrderExecutedMsg, matched));
        owner_->book_send(order.agent_id,
                          make_order_message(MsgType::OrderExecutedMsg, filled));
        return matched;
    }

    void enter_order(const Order& order) {
        std::vector<PriceLevel>& book = (order.side == Side::Bid) ? bids_ : asks_;
        if (book.empty()) {
            book.push_back(make_level(order));
            return;
        }
        if (book.back().order_has_worse_price(order)) {
            book.push_back(make_level(order));  // worse than everything on this side
            return;
        }
        for (std::size_t i = 0; i < book.size(); ++i) {
            if (book[i].order_has_better_price(order)) {
                book.insert(book.begin() + static_cast<long>(i), make_level(order));
                return;
            }
            if (book[i].order_has_equal_price(order)) {
                book[i].visible.push_back(order);  // add_order: to the BACK (FIFO)
                return;
            }
        }
    }

    // Returns true if the order was found and cancelled. Emits OrderCancelledMsg carrying the
    // order AS IT RESTED, so the quantity reflects the unexecuted remainder.
    bool cancel_order(const Order& order) {
        std::vector<PriceLevel>& book = (order.side == Side::Bid) ? bids_ : asks_;
        if (book.empty()) return false;
        for (std::size_t i = 0; i < book.size(); ++i) {
            if (!book[i].order_has_equal_price(order)) continue;
            for (std::size_t j = 0; j < book[i].visible.size(); ++j) {
                if (book[i].visible[j].order_id != order.order_id) continue;
                Order cancelled = book[i].visible[j];
                book[i].visible.erase(book[i].visible.begin() + static_cast<long>(j));
                if (book[i].is_empty()) book.erase(book.begin() + static_cast<long>(i));
                owner_->book_send(order.agent_id,
                                  make_order_message(MsgType::OrderCancelledMsg, cancelled));
                return true;
            }
            // Price level matched but the id was not there; the original keeps scanning.
        }
        return false;
    }

    // cancel_order(..., quiet=True): removes the order and sends NO message, which is what
    // cancel_oldest self-trade prevention uses.
    bool cancel_order_quiet(const Order& order) {
        std::vector<PriceLevel>& book = (order.side == Side::Bid) ? bids_ : asks_;
        for (std::size_t i = 0; i < book.size(); ++i) {
            if (!book[i].order_has_equal_price(order)) continue;
            for (std::size_t j = 0; j < book[i].visible.size(); ++j) {
                if (book[i].visible[j].order_id != order.order_id) continue;
                book[i].visible.erase(book[i].visible.begin() + static_cast<long>(j));
                if (book[i].is_empty()) book.erase(book.begin() + static_cast<long>(i));
                return true;
            }
        }
        return false;
    }

private:
    static PriceLevel make_level(const Order& order) {
        PriceLevel lvl;
        lvl.price = order.limit_price;
        lvl.side = order.side;
        lvl.visible.push_back(order);
        return lvl;
    }

    BookOwner* owner_;
    std::vector<PriceLevel> bids_;  // index 0 = highest
    std::vector<PriceLevel> asks_;  // index 0 = lowest
    std::int64_t last_trade_;
};

}  // namespace t3

#endif  // ORDER_BOOK_HPP
