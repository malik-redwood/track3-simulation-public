// test_parquet.cpp — write a file exercising every column shape the engine needs, so the
// Python side can verify it round-trips with the right dtypes.
//
// The case that matters: a nullable int64 holding ~1.6e18 nanosecond timestamps. If the pandas
// metadata is wrong it comes back as float64 and the values are silently corrupted.

#include <cstdio>
#include <string>
#include <vector>

#include "parquet.hpp"

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "test_out.parquet";

    const std::int64_t base = 1612483200000000050LL;  // a real t_send_ns from t3-s001
    std::vector<pq::Column> cols;

    pq::Column seq;
    seq.name = "seq"; seq.type = pq::Ty::Int64; seq.numpy_type = "int64";
    for (int i = 0; i < 7; ++i) seq.i64.push_back(i);
    cols.push_back(seq);

    // Nullable int64 with huge values: nulls at positions 0 and 4.
    pq::Column tsend;
    tsend.name = "t_send_ns"; tsend.type = pq::Ty::Int64; tsend.optional = true;
    tsend.numpy_type = "Int64";
    for (int i = 0; i < 7; ++i) {
        const bool null = (i == 0 || i == 4);
        tsend.i64.push_back(null ? 0 : base + i);
        tsend.present.push_back(null ? 0 : 1);
    }
    cols.push_back(tsend);

    pq::Column src;
    src.name = "src_id"; src.type = pq::Ty::Int32; src.numpy_type = "int32";
    for (int i = 0; i < 7; ++i) src.i32.push_back(i % 5);
    cols.push_back(src);

    pq::Column msg;
    msg.name = "msg_type"; msg.type = pq::Ty::String; msg.numpy_type = "string";
    for (const char* s : {"AGENT_WAKEUP", "LimitOrderMsg", "OrderExecutedMsg", "QUOTE_UPDATE",
                          "OrderAcceptedMsg", "MarketClosedMsg", "OrderCancelledMsg"})
        msg.str.emplace_back(s);
    cols.push_back(msg);

    // Nullable string-free nullable int: order_id, -1 encoded as null.
    pq::Column oid;
    oid.name = "order_id"; oid.type = pq::Ty::Int64; oid.optional = true;
    oid.numpy_type = "Int64";
    for (int i = 0; i < 7; ++i) {
        const bool null = (i % 3 == 0);
        oid.i64.push_back(null ? 0 : 1000 + i);
        oid.present.push_back(null ? 0 : 1);
    }
    cols.push_back(oid);

    // Nullable side: a string column with nulls.
    pq::Column side;
    side.name = "side"; side.type = pq::Ty::String; side.optional = true;
    side.numpy_type = "string";
    for (int i = 0; i < 7; ++i) {
        const bool null = (i == 2 || i == 5);
        side.str.emplace_back(null ? "" : (i % 2 ? "BID" : "ASK"));
        side.present.push_back(null ? 0 : 1);
    }
    cols.push_back(side);

    if (!pq::write(path, cols)) {
        std::fprintf(stderr, "write failed\n");
        return 1;
    }
    std::printf("wrote %s\n", path.c_str());
    return 0;
}
