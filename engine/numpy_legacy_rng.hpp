// numpy_legacy_rng.hpp — a bit-exact reimplementation of NumPy's legacy RandomState.
//
// WHY THIS HAS TO BE EXACT
//
// 42 of the 65 public Track-3 units are Tier A: identical row count, identical fill sequence,
// exact bidirectional event coverage, Kendall-tau >= 0.999. The trace is a deterministic
// function of the MT19937 draw sequence, so a single extra or missing 32-bit word anywhere
// changes message latencies, changes the order messages reach the exchange, and diverges the
// whole trace. A fast engine with a wrong RNG scores zero.
//
// Everything here mirrors numpy/random/src (mt19937.c) and
// numpy/random/src/distributions/random_distributions.c + legacy. The streams are validated
// against NumPy itself by test_rng.cpp / check_rng.py; NumPy guarantees RandomState stream
// stability across versions, so the contract is durable.
//
// MEASURED WORD COSTS (confirmed against numpy via RandomState.get_state()[2]):
//   draw_seed()               1 word   (randint(0, 2**32, dtype=uint64) returns the raw word)
//   randint(0, k)             1 word, +1 per rejection when k is not a power of two
//   random_sample()           2 words
//   uniform / exponential     2 words
//   pareto                    2 words
//   normal / lognormal        4 words on odd calls (+4 per polar rejection), 0 on even calls
//
// The `normal` cost being call-parity dependent AND rejection dependent is the single nastiest
// property here: you cannot precompute how many words a run will consume.

#ifndef NUMPY_LEGACY_RNG_HPP
#define NUMPY_LEGACY_RNG_HPP

#include <cmath>
#include <cstdint>

namespace nprng {

// --------------------------------------------------------------------------- MT19937
// numpy/random/src/mt19937/mt19937.c
class MT19937 {
public:
    static constexpr int kN = 624;
    static constexpr int kM = 397;

    MT19937() { seed(0u); }
    explicit MT19937(std::uint32_t s) { seed(s); }

    // mt19937_seed: identical to the reference init_genrand. numpy writes key[pos] first and
    // then advances the scalar with `+ pos + 1`, which lands on the same sequence as the
    // canonical `mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i`.
    void seed(std::uint32_t s) {
        s &= 0xffffffffu;
        for (int pos = 0; pos < kN; ++pos) {
            key_[pos] = s;
            s = 1812433253u * (s ^ (s >> 30)) + static_cast<std::uint32_t>(pos) + 1u;
        }
        pos_ = kN;
    }

    std::uint32_t next_uint32() {
        if (pos_ == kN) regenerate();
        std::uint32_t y = key_[pos_++];
        // Tempering.
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // mt19937_next_double: 27 bits from the first word, 26 from the second.
    double next_double() {
        const std::int32_t a = static_cast<std::int32_t>(next_uint32() >> 5);
        const std::int32_t b = static_cast<std::int32_t>(next_uint32() >> 6);
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    // Words consumed since the last buffer regeneration. Mirrors
    // RandomState.get_state()[2], which is what makes a wrong consumption count detectable
    // even when the returned value looks plausible.
    int word_pos() const { return pos_; }

private:
    void regenerate() {
        static constexpr std::uint32_t kMatrixA = 0x9908b0dfu;
        static constexpr std::uint32_t kUpper = 0x80000000u;
        static constexpr std::uint32_t kLower = 0x7fffffffu;
        std::uint32_t y;
        int i = 0;
        for (; i < kN - kM; ++i) {
            y = (key_[i] & kUpper) | (key_[i + 1] & kLower);
            key_[i] = key_[i + kM] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
        }
        for (; i < kN - 1; ++i) {
            y = (key_[i] & kUpper) | (key_[i + 1] & kLower);
            key_[i] = key_[i + (kM - kN)] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
        }
        y = (key_[kN - 1] & kUpper) | (key_[0] & kLower);
        key_[kN - 1] = key_[kM - 1] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
        pos_ = 0;
    }

    std::uint32_t key_[kN]{};
    int pos_ = kN;
};

// --------------------------------------------------------------------------- RandomState
class LegacyRandomState {
public:
    LegacyRandomState() = default;
    explicit LegacyRandomState(std::uint32_t s) : mt_(s) {}

    void seed(std::uint32_t s) {
        mt_.seed(s);
        has_gauss_ = false;
        gauss_ = 0.0;
    }

    // --- raw access -------------------------------------------------------
    std::uint32_t next_uint32() { return mt_.next_uint32(); }
    double random_sample() { return mt_.next_double(); }

    int word_pos() const { return mt_.word_pos(); }
    bool has_gauss() const { return has_gauss_; }
    double cached_gauss() const { return gauss_; }

    // --- bounded integers -------------------------------------------------
    // numpy's random_bounded_uint64 short-circuits to the 32-bit generator whenever the range
    // fits in 32 bits ("Call 32-bit generator if range in 32-bit"), which is why a uint64
    // randint over [0, 2**32) costs ONE word rather than two.
    static std::uint32_t gen_mask(std::uint32_t rng) {
        std::uint32_t mask = rng;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        return mask;
    }

    // buffered_bounded_masked_uint32: masked rejection sampling.
    std::uint32_t bounded_masked_uint32(std::uint32_t rng, std::uint32_t mask) {
        std::uint32_t val;
        while ((val = (mt_.next_uint32() & mask)) > rng) {
            // rejection; consumes another word
        }
        return val;
    }

    // `randint(low, high)` with high EXCLUSIVE, as numpy's randint is.
    std::int64_t randint(std::int64_t low, std::int64_t high) {
        const std::uint64_t span = static_cast<std::uint64_t>(high - low - 1);
        if (span == 0) return low;  // numpy returns `off` when rng == 0, drawing nothing
        const std::uint32_t rng = static_cast<std::uint32_t>(span);
        return low + static_cast<std::int64_t>(bounded_masked_uint32(rng, gen_mask(rng)));
    }

    // The sub-seed draw used throughout abides_fork/config.py:
    //     np.random.RandomState(seed=np.random.randint(low=0, high=2**32, dtype="uint64"))
    // rng == mask == 0xffffffff, so the rejection loop can never fire and this is exactly one
    // tempered word returned verbatim.
    std::uint32_t draw_seed() {
        return bounded_masked_uint32(0xffffffffu, 0xffffffffu);
    }

    // --- continuous distributions ----------------------------------------
    // legacy_gauss: Marsaglia polar, caching the second variate. Returns f*x2 and keeps f*x1 —
    // the order is load-bearing.
    double gauss() {
        if (has_gauss_) {
            const double t = gauss_;
            has_gauss_ = false;
            gauss_ = 0.0;
            return t;
        }
        double x1, x2, r2;
        do {
            x1 = 2.0 * mt_.next_double() - 1.0;
            x2 = 2.0 * mt_.next_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_ = f * x1;
        has_gauss_ = true;
        return f * x2;
    }

    double normal(double loc, double scale) { return loc + scale * gauss(); }

    // legacy_standard_exponential: -log(1 - U), since U is [0, 1).
    double standard_exponential() { return -std::log(1.0 - mt_.next_double()); }
    double exponential(double scale) { return scale * standard_exponential(); }

    double pareto(double a) { return std::exp(standard_exponential() / a) - 1.0; }
    double lognormal(double mean, double sigma) { return std::exp(normal(mean, sigma)); }

    // random_uniform(low, range) = low + range * next_double
    double uniform(double low, double high) { return low + (high - low) * mt_.next_double(); }

private:
    MT19937 mt_;
    bool has_gauss_ = false;
    double gauss_ = 0.0;
};

// --------------------------------------------------------------------------- helpers
// Python's round() is round-half-to-EVEN, and the latency model does
// `int(round(float(np.clip(v, lo, hi))))`. std::nearbyint under the default FE_TONEAREST is
// also round-half-to-even, so this matches; std::round (half away from zero) would NOT.
inline std::int64_t py_round_to_i64(double v) {
    return static_cast<std::int64_t>(std::nearbyint(v));
}

inline double clip(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace nprng

#endif  // NUMPY_LEGACY_RNG_HPP
