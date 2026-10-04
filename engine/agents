// agents.hpp — the four Track-3 agent types from abides_fork/agents.py.
//
// NoiseTrader is the only one that draws randomness: exactly three draws per act(), in this
// order, verified bit-exact against NumPy and against every NoiseTrader in all 65 units.
//     size   = int(max(1, round(normal(order_size_mean, order_size_std))))
//     buy    = bool(randint(0, 2))
//     offset = int(randint(0, price_offset_ticks + 1))
//
// MarketMaker and MomentumTrader draw nothing. ValueTrader draws once via
// oracle.observe_price(..., sigma_n), but ONLY if the book is not two-sided-empty -- the early
// return makes its stream position depend on book state.

#ifndef AGENTS_HPP
#define AGENTS_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "exchange_agent.hpp"
#include "oracle.hpp"

namespace t3 {

class NoiseTrader : public ScheduledAgent {
public:
    NoiseTrader(int id, nprng::LegacyRandomState rs, int ex, std::int64_t interval,
                double size_mean, double size_std, int offset_ticks, std::int64_t ref_price)
        : ScheduledAgent(id, rs, ex, interval), size_mean_(size_mean), size_std_(size_std),
          offset_ticks_(offset_ticks), ref_price_(ref_price) {}

    void act() override {
        const std::int64_t size = std::max<std::int64_t>(
            1, nprng::py_round_to_i64(random_state_.normal(size_mean_, size_std_)));
        const bool buy = random_state_.randint(0, 2) != 0;
        const std::int64_t off = random_state_.randint(0, offset_ticks_ + 1);
        // Anchor on the opposite touch, falling back through the near touch to the reference.
        // These are truthiness tests in Python, so a price of 0 also falls through.
        if (buy) {
            const std::int64_t anchor =
                known_ask_ ? known_ask_ : (known_bid_ ? known_bid_ : ref_price_);
            place_limit_order(size, Side::Bid, anchor + off);
        } else {
            const std::int64_t anchor =
                known_bid_ ? known_bid_ : (known_ask_ ? known_ask_ : ref_price_);
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
        const std::int64_t mid =
            (known_bid_ && known_ask_) ? ((known_bid_ + known_ask_) / 2) : ref_price_;
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

// ValueTrader: one oracle observation per act, on the AGENT's stream, and an early return that
// draws nothing when the book has neither side.
class ValueTrader : public ScheduledAgent {
public:
    ValueTrader(int id, nprng::LegacyRandomState rs, int ex, std::int64_t interval,
                double size_mean, int threshold_ticks, double sigma_n,
                SparseMeanRevertingOracle* oracle)
        : ScheduledAgent(id, rs, ex, interval), size_mean_(size_mean),
          threshold_ticks_(threshold_ticks), sigma_n_(sigma_n), oracle_(oracle) {}

    void act() override {
        double mid;
        if (known_bid_ && known_ask_) {
            mid = (static_cast<double>(known_bid_) + static_cast<double>(known_ask_)) / 2.0;
        } else if (known_bid_) {
            mid = static_cast<double>(known_bid_);
        } else if (known_ask_) {
            mid = static_cast<double>(known_ask_);
        } else {
            return;  // no draw at all
        }
        const std::int64_t fundamental =
            oracle_->observe_price(current_time_, random_state_, sigma_n_);
        const std::int64_t size = std::max<std::int64_t>(1, nprng::py_round_to_i64(size_mean_));
        if (mid < static_cast<double>(fundamental - threshold_ticks_) && known_ask_) {
            place_limit_order(size, Side::Bid, known_ask_);
        } else if (mid > static_cast<double>(fundamental + threshold_ticks_) && known_bid_) {
            place_limit_order(size, Side::Ask, known_bid_);
        }
    }

private:
    double size_mean_;
    int threshold_ticks_;
    double sigma_n_;
    SparseMeanRevertingOracle* oracle_;
};

// MomentumTrader: no randomness; compares the mid against the mid `lookback` wakeups ago.
class MomentumTrader : public ScheduledAgent {
public:
    MomentumTrader(int id, nprng::LegacyRandomState rs, int ex, std::int64_t interval,
                   double size_mean, int threshold_ticks, int lookback)
        : ScheduledAgent(id, rs, ex, interval), size_mean_(size_mean),
          threshold_ticks_(threshold_ticks), lookback_(lookback < 1 ? 1 : lookback) {}

    void act() override {
        double mid;
        if (known_bid_ && known_ask_) {
            mid = (static_cast<double>(known_bid_) + static_cast<double>(known_ask_)) / 2.0;
        } else if (known_bid_) {
            mid = static_cast<double>(known_bid_);
        } else if (known_ask_) {
            mid = static_cast<double>(known_ask_);
        } else {
            return;
        }
        mid_history_.push_back(mid);
        if (static_cast<int>(mid_history_.size()) > lookback_ + 1) mid_history_.erase(mid_history_.begin());
        if (static_cast<int>(mid_history_.size()) <= lookback_) return;
        const double past = mid_history_.front();
        const std::int64_t size = std::max<std::int64_t>(1, nprng::py_round_to_i64(size_mean_));
        if (mid > past + threshold_ticks_ && known_ask_) {
            place_limit_order(size, Side::Bid, known_ask_);
        } else if (mid < past - threshold_ticks_ && known_bid_) {
            place_limit_order(size, Side::Ask, known_bid_);
        }
    }

private:
    double size_mean_;
    int threshold_ticks_;
    int lookback_;
    std::vector<double> mid_history_;
};

}  // namespace t3

#endif  // AGENTS_HPP
