#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

echo "=== 1. Environment & Repository Check ==="
cd /workspaces/track3-simulation-public
pwd
echo "Units found: $(ls units | wc -l)"
if [ -f "check_candidate.py" ]; then echo "check_candidate.py OK"; fi

if docker info > /dev/null 2>&1; then
    echo "=== Docker OK ==="
else
    echo "=== NO DOCKER (Skipping container checks) ==="
fi

echo "=== 2. Installing Dependencies ==="
python3 -m pip install --user ziglang
python3 -m pip install --user "qfbench2-common[data] @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.5.1#subdirectory=common"
python3 -m pip install --user -e .

if command -v git-lfs &> /dev/null; then
    git lfs pull
fi

echo "=== 3. Verifying Python Imports & Zig ==="
python3 -c "import qfbench2_common, qfbench2_track_simulation, pyarrow, pandas; print('imports OK')"
python3 -m ziglang version

echo "=== 4. Validating Sweep REPO Path ==="
python3 -c "import sys;sys.path.insert(0, 'engine');from sweep import REPO;print('REPO =',REPO,'| units found:',(REPO/'units').is_dir())"

echo "=== 5. Compiling C++ Engine inside engine/ ==="
cd engine
rm -rf zig-cache ~/.cache/zig

# Compile run_unit (using clang++ fallback if zig wrapper fails, or zig c++)
python3 -m ziglang c++ -std=c++20 -O2 -Wno-everything -ffp-contract=off -o run_unit run_unit.cpp kernel.cpp || \
    clang++ -std=c++20 -O2 -Wno-everything -ffp-contract=off -o run_unit run_unit.cpp kernel.cpp

# Compile static musl Linux simulator binary
python3 -m ziglang c++ -std=c++20 -O2 -s -Wno-everything -ffp-contract=off -target x86_64-linux-musl -static -o simulate_linux simulate.cpp kernel.cpp || \
    clang++ -std=c++20 -O2 -Wno-everything -ffp-contract=off -o simulate_linux simulate.cpp kernel.cpp

chmod +x simulate_linux validate_image.sh

echo "=== 6. Running Preflight & Sweep ==="
python3 preflight.py simulate_linux || python preflight.py simulate_linux
python3 sweep.py ./run_unit

echo "=== 7. Final Validation Check ==="
./run_unit --version || true
cd /workspaces/track3-simulation-public
python3 check_candidate.py --candidate engine/run_outputs

echo "=== All steps completed successfully! ==="
