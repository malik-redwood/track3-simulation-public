#!/usr/bin/env bash
# Build + validate the Track-3 RNG core.
#
# Works with either a real clang/g++ or with `python -m ziglang c++` (pip install ziglang),
# which is how this gets built on the Windows dev box where no compiler is installable.
#
#     bash build.sh              # native build + validate against the NumPy fixtures
#     bash build.sh linux        # also cross-compile the static x86_64-linux-musl binary
set -euo pipefail

# ---------------------------------------------------------------------------
# -ffp-contract=off IS MANDATORY, NOT AN OPTIMISATION PREFERENCE.
#
# Without it the compiler fuses `loc + scale * gauss()` into a single fused multiply-add.
# FMA keeps more intermediate precision than a separate multiply and add, so the result differs
# from NumPy by one unit in the last place. Measured: 9 of 930 fixture checks failed with
# 1-ULP errors in `normal` and `uniform`, every one of which disappeared with contraction off.
#
# A 1-ULP error cannot change the DRAW COUNT (the polar rejection test `r2 >= 1.0` uses only
# exact IEEE multiply/add on next_double output, never libm), so it would not desynchronise the
# stream. It can still flip an `int(round(x))` at a rounding boundary, which is enough to break
# a Tier-A trace. Do not remove this flag.
# ---------------------------------------------------------------------------
FLAGS=(-std=c++20 -O2 -ffp-contract=off -Wall -Wextra)

# PY: the interpreter that has numpy + pyarrow (needed to generate fixtures and read ledgers).
# CXX_CMD: optional explicit compiler, e.g. on the Windows dev box where ziglang lives in a venv:
#   PY=/c/Users/.../.venv-t3-13/Scripts/python.exe CXX_CMD="$PY -m ziglang c++" bash build.sh
PY="${PY:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
fi

if [ -n "${CXX_CMD:-}" ]; then
    read -r -a CXX <<< "$CXX_CMD"
elif command -v clang++ >/dev/null 2>&1; then
    CXX=(clang++)
elif command -v g++ >/dev/null 2>&1; then
    CXX=(g++)
elif "$PY" -m ziglang version >/dev/null 2>&1; then
    CXX=("$PY" -m ziglang c++)
else
    echo "no C++ compiler found. Options:" >&2
    echo "  pip install ziglang          (then re-run; works with no admin rights)" >&2
    echo "  CXX_CMD='clang++' bash build.sh" >&2
    exit 1
fi
echo "compiler: ${CXX[*]}"
echo "python:   $PY"

echo "--- native build ---"
"${CXX[@]}" "${FLAGS[@]}" -o test_rng test_rng.cpp

echo "--- regenerate fixtures from NumPy ---"
"$PY" gen_fixtures.py > fixtures.json

echo "--- validate against NumPy (expect 930/930) ---"
"$PY" check_rng.py ./test_rng

echo "--- validate against the 65 shipped reference ledgers (expect 65/65) ---"
"$PY" verify_units_cpp.py ./test_rng | tail -3

echo "--- validate the oracle against the real patched ABIDES module (expect 13/13) ---"
"$PY" check_oracle.py ./test_rng | tail -3

echo "--- build the simulator ---"
"${CXX[@]}" "${FLAGS[@]}" -o test_bootstrap test_bootstrap.cpp kernel.cpp

echo "--- t3-s001 message ledger: expect all 664 rows identical ---"
"$PY" check_bootstrap.py ./test_bootstrap --rows 700 | tail -3

echo "--- t3-s001 trace.parquet: expect all 604 rows identical ---"
"$PY" check_trace.py ./test_bootstrap | tail -3

echo "--- grade t3-s001 through the real Tier-A gate: expect 1/1 ---"
"$PY" emit_unit.py ./test_bootstrap | tail -4

if [ "${1:-}" = "linux" ]; then
    echo "--- cross-compile the SUBMISSION binary (static x86_64-linux-musl, stripped) ---"
    "${CXX[@]}" "${FLAGS[@]}" -s -target x86_64-linux-musl -static         -o simulate_linux simulate.cpp kernel.cpp
    ls -la simulate_linux

    echo "--- pre-flight: static, shell-free, linux/amd64 (no Docker needed) ---"
    "$PY" preflight.py simulate_linux

    "${CXX[@]}" "${FLAGS[@]}" -target x86_64-linux-musl -static -o test_rng_linux test_rng.cpp
    ls -la test_rng_linux
    cat <<'NOTE'

NOTE: the Linux binary is NOT validated by this script when cross-compiling from Windows --
it cannot be executed here. Run on Linux:
    chmod +x test_rng_linux
    python gen_fixtures.py > fixtures.json     # regenerate with the TARGET platform's numpy
    python check_rng.py ./test_rng_linux
    python verify_units_cpp.py ./test_rng_linux
This matters because std::log / std::exp come from libm, which is not required to be
correctly rounded and differs between MSVCRT, glibc and musl. See README.md "libm".
NOTE
fi
