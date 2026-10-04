// kernel.hpp — the ABIDES discrete-event kernel, reproduced exactly.
//
// Mirrors abides-core/abides_core/kernel.py at pinned commit f9cbe51 with
// kernel_message_ledger.patch applied.
//
// THE FOUR THINGS THAT DECIDE THE TRACE
//
// 1. TOTAL ORDER. Python pushes `(deliver_at, (sender_id, recipient_id, message))` onto a
//    queue.PriorityQueue, and tuple comparison falls through left to right, with Message.__lt__
//    comparing message_id. So the order is
//        (deliver_at, sender_id, recipient_id, message_id)
//    Verified against the t3-s001 ledger: four wakeups delivered at the same instant carrying
//    ids 17, 22, 19, 21 arrive in src_id order 1,2,3,4 — so sender_id really does outrank
//    message_id.
//
// 2. message_id IS A GLOBAL CONSTRUCTION COUNTER starting at 1, shared by every Message
//    subclass, incremented when the object is CONSTRUCTED (not sent). simulate.py resets it to 1
//    per run. Every object that is built must therefore be built in the same order, including
//    ones that are never sent, and a broadcast must reuse one object across recipients.
//
// 3. REQUEUE WHEN THE AGENT IS IN THE FUTURE. If agent_current_times[recipient] > current_time
//    the SAME message object is pushed back at the agent's availability time and the ledger is
//    NOT written. Only the final delivery is recorded, so requeues are invisible in the ledger
//    but do reorder everything.
//
// 4. COMPUTATION-DELAY ASYMMETRY. On the wakeup branch the agent clock advances AFTER
//    wakeup() returns; on the message branch it advances BEFORE receive_message(), and once per
//    sub-message of a batch. send_message is unaffected because it reads kernel current_time,
//    not the agent clock.

#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "messages.hpp"
#include "numpy_legacy_rng.hpp"

namespace t3 {

class Agent;

// --------------------------------------------------------------------------- latency
// config.py ScenarioLatencyModel. Returns 0 and draws NOTHING when sender == recipient.
class LatencyModel {
public:
    enum class Kind { Deterministic, Uniform, LogNormal, Pareto };

    LatencyModel() = default;
    LatencyModel(Kind kind, double mean_ns, double sigma, double min_ns, double max_ns,
                 double alpha, std::uint32_t seed)
        : kind_(kind), mean_ns_(mean_ns), sigma_(sigma), min_ns_(min_ns), max_ns_(max_ns),
          alpha_(alpha), rs_(seed) {
        mu_ = mean_ns_ > 0.0 ? std::log(mean_ns_) : 0.0;
    }

    std::int64_t get_latency(int sender_id, int recipient_id) {
        if (sender_id == recipient_id) return 0;  // no draw
        double v;
        switch (kind_) {
            case Kind::LogNormal: v = rs_.lognormal(mu_, sigma_); break;
            case Kind::Uniform:   v = rs_.uniform(min_ns_, max_ns_); break;
            case Kind::Pareto:    v = (min_ns_ > 0.0 ? min_ns_ : 1.0) * (1.0 + rs_.pareto(alpha_));
                                  break;
            default:              v = mean_ns_; break;
        }
        return nprng::py_round_to_i64(nprng::clip(v, min_ns_, max_ns_));
    }

private:
    Kind kind_ = Kind::Deterministic;
    double mean_ns_ = 0.0, sigma_ = 0.0, min_ns_ = 0.0, max_ns_ = 1e12, alpha_ = 1.5, mu_ = 0.0;
    nprng::LegacyRandomState rs_;
};

// --------------------------------------------------------------------------- ledger
struct LedgerRow {
    std::int64_t seq = -1;          // delivery order, joined on afterwards
    std::int64_t message_id = 0;
    int src_id = 0;
    int dst_id = 0;
    bool has_send = false;        // wakeups carry no send time
    std::int64_t t_send_ns = 0;
    std::int64_t t_recv_ns = 0;
    std::int64_t latency_ns = 0;
    MsgType msg_type = MsgType::WakeupMsg;
    std::int64_t order_id = -1;   // -1 == null
    std::int64_t causal_parent = -1;
};

// --------------------------------------------------------------------------- queue entry
struct QueueEntry {
    std::int64_t deliver_at;
    int sender_id;
    int recipient_id;
    std::int64_t message_id;
    MessagePtr message;

    // std::priority_queue is a MAX-heap, so invert: "less" means lower priority.
    bool operator<(const QueueEntry& o) const {
        if (deliver_at != o.deliver_at) return deliver_at > o.deliver_at;
        if (sender_id != o.sender_id) return sender_id > o.sender_id;
        if (recipient_id != o.recipient_id) return recipient_id > o.recipient_id;
        return message_id > o.message_id;
    }
};

// --------------------------------------------------------------------------- kernel
class Kernel {
public:
    Kernel(std::int64_t start_time, std::int64_t stop_time, int default_computation_delay)
        : start_time_(start_time), stop_time_(stop_time), current_time_(start_time),
          default_computation_delay_(default_computation_delay) {}

    void set_latency_model(LatencyModel model) { latency_ = std::move(model); }

    // Agents are owned by the caller and registered in construction order; index == agent id.
    void add_agent(Agent* a) {
        agents_.push_back(a);
        agent_current_times_.push_back(start_time_);
        agent_computation_delays_.push_back(default_computation_delay_);
    }

    std::int64_t current_time() const { return current_time_; }
    std::size_t n_agents() const { return agents_.size(); }
    const std::vector<LedgerRow>& ledger() const { return ledger_; }

    // trace.py keeps only DELIVERED messages and sorts them by the kernel's processing order.
    // Rows are appended at SEND time, so the delivery seq has to be joined on afterwards --
    // keyed on (message_id, recipient), because one broadcast object is delivered N times.
    std::vector<LedgerRow> delivered_ledger_sorted() const {
        std::vector<LedgerRow> out;
        out.reserve(ledger_.size());
        for (const LedgerRow& r : ledger_) {
            auto it = deliver_seq_by_key_.find({r.message_id, r.dst_id});
            if (it == deliver_seq_by_key_.end()) continue;  // sent but never delivered
            LedgerRow copy = r;
            copy.seq = it->second;
            out.push_back(copy);
        }
        std::sort(out.begin(), out.end(),
                  [](const LedgerRow& a, const LedgerRow& b) { return a.seq < b.seq; });
        return out;
    }

    void set_computation_delay(int agent_id, int delay) {
        agent_computation_delays_[agent_id] = delay;
    }

    // Kernel.set_wakeup: constructs a FRESH WakeupMsg, so every wakeup burns a message id.
    void set_wakeup(int sender_id, std::int64_t requested_time) {
        MessagePtr m = make_message(MsgType::WakeupMsg);
        queue_.push(QueueEntry{requested_time, sender_id, sender_id, m->message_id, m});
    }

    // Kernel.send_message. One latency draw per call; a broadcast is N calls sharing one object.
    void send_message(int sender_id, int recipient_id, const MessagePtr& message, int delay = 0) {
        const std::int64_t sent_time = current_time_ + agent_computation_delays_[sender_id] +
                                       current_agent_additional_delay_ + delay;
        const std::int64_t latency = latency_.get_latency(sender_id, recipient_id);
        const std::int64_t deliver_at = sent_time + latency;
        queue_.push(QueueEntry{deliver_at, sender_id, recipient_id, message->message_id, message});

        LedgerRow row;
        row.message_id = message->message_id;
        row.src_id = sender_id;
        row.dst_id = recipient_id;
        row.has_send = true;
        row.t_send_ns = sent_time;
        row.t_recv_ns = deliver_at;
        row.latency_ns = deliver_at - sent_time;
        row.msg_type = message->type;
        row.order_id = message->order_id;
        row.causal_parent = current_causal_uid_;
        ledger_.push_back(row);
    }

    void delay_agent(int /*sender_id*/, std::int64_t additional) {
        current_agent_additional_delay_ += additional;
    }

    void run();

private:
    std::int64_t start_time_;
    std::int64_t stop_time_;
    std::int64_t current_time_;
    int default_computation_delay_;

    std::vector<Agent*> agents_;
    std::vector<std::int64_t> agent_current_times_;
    std::vector<int> agent_computation_delays_;
    std::int64_t current_agent_additional_delay_ = 0;

    std::priority_queue<QueueEntry> queue_;
    LatencyModel latency_;

    std::vector<LedgerRow> ledger_;
    std::map<std::pair<std::int64_t, int>, std::int64_t> deliver_seq_by_key_;
    std::int64_t deliver_seq_ = 0;
    std::int64_t current_causal_uid_ = -1;
};

}  // namespace t3

#endif  // KERNEL_HPP
