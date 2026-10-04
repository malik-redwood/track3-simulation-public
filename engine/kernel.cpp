// kernel.cpp — Kernel::run, the event loop from abides-core/abides_core/kernel.py::runner.

#include "kernel.hpp"

#include "agent.hpp"

namespace t3 {

void Kernel::run() {
    // Kernel.initialize: kernel_initializing for ALL agents, then kernel_starting for ALL
    // agents, each in id order. The exchange's mkt_close wakeup is created in the FIRST loop,
    // which is why it takes message_id 1 -- ahead of every start-time wakeup.
    for (Agent* a : agents_) a->attach(this);
    for (Agent* a : agents_) a->kernel_initializing();
    for (Agent* a : agents_) a->kernel_starting(start_time_);

    current_time_ = start_time_;

    while (!queue_.empty() && current_time_ <= stop_time_) {
        const QueueEntry entry = queue_.top();
        queue_.pop();
        current_time_ = entry.deliver_at;

        const int sender_id = entry.sender_id;
        const int recipient_id = entry.recipient_id;
        const MessagePtr message = entry.message;

        // Reset between messages, before any dispatch.
        current_agent_additional_delay_ = 0;

        if (message->type == MsgType::WakeupMsg) {
            // Agent in the future: requeue the SAME object and do not touch the ledger.
            if (agent_current_times_[recipient_id] > current_time_) {
                queue_.push(QueueEntry{agent_current_times_[recipient_id], sender_id,
                                       recipient_id, message->message_id, message});
                continue;
            }
            agent_current_times_[recipient_id] = current_time_;

            // Ledger + causal root recorded on FINAL delivery only. The wakeup becomes the
            // causal parent of whatever the agent sends in response.
            current_causal_uid_ = message->message_id;
            deliver_seq_by_key_[{message->message_id, recipient_id}] = deliver_seq_++;
            LedgerRow row;
            row.message_id = message->message_id;
            row.src_id = recipient_id;
            row.dst_id = recipient_id;
            row.has_send = false;  // wakeups carry no send time
            row.t_recv_ns = current_time_;
            row.latency_ns = 0;
            row.msg_type = MsgType::WakeupMsg;
            ledger_.push_back(row);

            agents_[recipient_id]->wakeup(current_time_);

            // Wakeup branch: the agent clock advances AFTER wakeup() returns.
            agent_current_times_[recipient_id] +=
                agent_computation_delays_[recipient_id] + current_agent_additional_delay_;

        } else {
            if (agent_current_times_[recipient_id] > current_time_) {
                queue_.push(QueueEntry{agent_current_times_[recipient_id], sender_id,
                                       recipient_id, message->message_id, message});
                continue;
            }
            agent_current_times_[recipient_id] = current_time_;

            // Message branch: the agent clock advances BEFORE receive_message().
            agent_current_times_[recipient_id] +=
                agent_computation_delays_[recipient_id] + current_agent_additional_delay_;

            current_causal_uid_ = message->message_id;
            deliver_seq_by_key_[{message->message_id, recipient_id}] = deliver_seq_++;

            agents_[recipient_id]->receive_message(current_time_, sender_id, message);
        }
    }

    for (Agent* a : agents_) a->kernel_stopping();
}

}  // namespace t3
