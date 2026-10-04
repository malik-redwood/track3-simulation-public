# Engine — the Track-3 simulator

**Status: correctness-complete on the public roster. 71/71 units pass the real gate.**

```
930/930   generator checks vs NumPy, bit-exact
65/65     units reproduced from the shipped reference ledgers (7,115,390 values)
13/13     oracle cases vs the real patched ABIDES module
664/664   t3-s001 message-ledger rows identical
604/604   t3-s001 trace rows identical
71/71     units PASS the gate, with Parquet written entirely from C++
```

Throughput: arithmetic mean **1,290,215 events/sec** (91x the 14,159 baseline); the whole
roster simulates in 6.5 s against the baseline's 582 s.

## Submission artifacts

| file | what |
|---|---|
| `simulate.cpp` | the entry point: `simulate` and `simulate-batch` |
| `parquet.hpp` | Parquet writer, no pyarrow (importing it costs ~500 ms = 4.27x on the score) |
| `sha256.hpp` | `trace_sha256` for `events.json` |
| `Dockerfile` | `FROM scratch` + one 633 KB static binary |
| `validate_image.sh` | builds the image, runs every unit under `--network none`, times containers the organizer's way, grades the outputs |

Build the submission binary (from the dev box, cross-compiled):

```bash
python -m ziglang c++ -std=c++20 -O2 -s -ffp-contract=off \
    -target x86_64-linux-musl -static -o simulate_linux simulate.cpp kernel.cpp
docker build --platform=linux/amd64 -t t3-sim:latest .
```

**Not yet verified:** the Linux binary has never been executed and the image has never been
built — no Docker on the dev box. `validate_image.sh` closes that gap in one command.

---

## The RNG surface

```
numpy_legacy_rng.hpp    NumPy legacy RandomState (header-only, no dependencies)
oracle.hpp              SparseMeanRevertingOracle, incl. megashocks + scheduled jumps
test_rng.cpp            spec-driven driver, emits results as IEEE-754 bit patterns
gen_fixtures.py         ground truth from NumPy's RandomState
check_rng.py            C++ vs NumPy, bit for bit
verify_units_cpp.py     C++ vs the 65 shipped reference ledgers
check_oracle.py         C++ vs the REAL patched ABIDES oracle module
build.sh                build + run all three validations
```

## Validation

| test | result |
|---|---|
| `check_rng.py` — 20 cases vs NumPy, raw bit patterns | **930/930 exact** |
| `verify_units_cpp.py` — latency streams vs the shipped reference ledgers | **65/65 units**, 7,115,390 values |
| `check_oracle.py` — vs the real patched ABIDES oracle | **13/13 cases**, 1,088 observations |

The ledger test is the most meaningful. It reproduces realized message latencies recorded by the
pinned ABIDES baseline on Linux, months ago, on hardware we do not have — driving the whole chain
from C++: scenario seed → global MT19937 → `N+2` sub-seed draws around the oracle's global
`exponential` → latency sub-seed → per-message latency stream. 7.1 million integer latency values
across 65 units, all matching.

`check_oracle.py` imports the **actual** module from the pinned checkout rather than testing
against my reading of it (`abides_core` is stubbed to `NanosecondTime = int`, since its real
`__init__` pulls in the kernel). The megashock loop genuinely executes — 5, 7, 16 and 138
iterations across the four megashock cases — and the scheduled-jump patch is consumed in three
more, so the cases are not passing trivially.

## Build requirement: `-ffp-contract=off`

**Mandatory.** Without it the compiler fuses `loc + scale * gauss()` into a fused multiply-add,
which retains more intermediate precision than a separate multiply and add, and the result differs
from NumPy by 1 ULP. Measured: 9 of 930 fixture checks failed with last-hex-digit errors in
`normal` and `uniform`; all 9 disappeared with contraction disabled.

A 1-ULP error cannot desynchronise the draw stream — the polar rejection test `r2 >= 1.0 || r2 == 0.0`
uses only exact IEEE multiply/add on `next_double` output and never touches libm, so rejection
behaviour is bit-stable across platforms and compilers. But it can flip an `int(round(x))` at a
boundary, and that is enough to break a Tier-A trace.

## Measured word costs

Confirmed against `RandomState.get_state()[2]`, which is why the fixtures assert the word position
after every call and not just the value — a wrong consumption count produces plausible-looking
values that a value-only check would pass.

| call | 32-bit words |
|---|---|
| `draw_seed()` (`randint(0, 2**32, dtype=uint64)`) | **1**, returned verbatim |
| `randint(0, k)` | 1, **+1 per rejection** when `k` is not a power of two |
| `random_sample`, `uniform`, `exponential`, `pareto` | 2 |
| `normal` / `lognormal` | **4 on odd calls (+4 per polar rejection), 0 on even calls** |

The `normal` cost is both call-parity dependent and rejection dependent, so the number of words a
run consumes cannot be precomputed. Draws must be pulled lazily in event order.

`randint(0, 2**32, dtype=uint64)` costing one word rather than two is NumPy's
`random_bounded_uint64` short-circuit: "Call 32-bit generator if range in 32-bit". With
`rng == mask == 0xffffffff` the rejection loop can never fire, so it is exactly one tempered word.

## libm

`std::log` and `std::exp` are not required by IEEE-754 to be correctly rounded, and they differ
between MSVCRT, glibc and musl. NumPy uses the platform libm too, so the fixtures here are
Windows-flavoured while the reference traces were produced by NumPy 1.26.4 on Linux/glibc.

Empirically this does not matter: all 7,115,390 latency values match, and each one is
`int(round(clip(exp(normal(...)))))` — i.e. it passes through both `exp` and a rounding step. If
MSVCRT and glibc disagreed materially on `exp`, these would not all agree.

It remains worth re-running both validations natively on Linux before shipping, because the
argument above is empirical rather than a proof. `build.sh linux` cross-compiles the static
musl binary but deliberately does not claim to have validated it.

## Toolchain note

No C++ compiler is installable on the dev box (no admin, winget blocked by Group Policy), so this
builds with `pip install ziglang` — a clang-based C/C++ compiler that ships as a Python wheel and
cross-compiles to `x86_64-linux-musl` static. That solves both local iteration and the eventual
submission binary. `build.sh` auto-detects clang/g++/ziglang.

## Python's numeric tower is load-bearing in the oracle

`mst = self.mkt_open + ms_time_delta` is `int + float = float`, and nanosecond timestamps are
~1.6e18 — past 2^53, where a double stops representing every integer (granularity there is 256 ns).
So the oracle's cached `pt` is an **exact int** until a megashock fires and a **float** afterwards,
which changes how `d = ts - pt` is computed, and `mst < current_time` is a float-vs-int comparison
that CPython performs **exactly** rather than by coercing the int to double.

Doing all of that in `double` would quantise timestamps to 256 ns and diverge. `PyNum` in
`oracle.hpp` reproduces the tower: exact int arithmetic while both operands are ints, exact
mixed-mode comparison via `lt_double_int`.

This is not an edge case — megashocks are live in **13 of the 65 public units**, including all six
reactive-agent units, which are Tier A.

Two smaller traps in the same file: `mst = pt + int(exponential)` **truncates** the draw (the
constructor's first megashock does not), and `megashock_mean = float(jump_sigma) or 1000.0` turns a
configured `0.0` into `1000.0` because `0.0` is falsy in Python.

## Next

1. **Kernel event loop** — priority queue on `(deliver_at, sender_id, recipient_id, message_id)`,
   global `message_id` from 1, the requeue-when-in-future rule, the wakeup/message
   computation-delay asymmetry.
2. **Order book + the four agents**, then reproduce `t3-s001` end to end: 604 events, the smallest
   unit, and a complete Tier-A test.
3. **Parquet writer** in compiled code — importing pyarrow costs ~500 ms, which is 4.27x on the
   ranking score.
