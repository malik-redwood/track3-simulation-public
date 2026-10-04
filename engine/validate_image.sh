#!/usr/bin/env bash
# Validate the submission image the way the organizer will: run it per unit under the real
# constraints, then grade the outputs with the gate.
#
# Run this in a Linux environment with Docker (e.g. the Codespace). From the engine directory,
# with the repo checkout alongside.
#
#     bash validate_image.sh [path-to-track3-simulation-public]
#
# What it checks, in order:
#   1. the image builds and the binary runs in a scratch container (no shell, no libc)
#   2. every unit produces outputs under --network none with the card's CPU/memory caps
#   3. the gate passes on those outputs
#   4. container wall clock per unit, measured the organizer's way, giving the real events/sec
set -euo pipefail

REPO="${1:-../../track3-simulation-public}"
IMAGE="t3-sim:latest"
OUT="$(pwd)/image_outputs"
CAPS=(--network none --cpus "${CPUS:-4}" --memory "${MEM:-16g}")

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
[ -f simulate_linux ] || { echo "simulate_linux missing -- cross-compile it first"; exit 1; }

echo "=== 1. build ==="
docker build --platform=linux/amd64 -q -t "$IMAGE" . >/dev/null
echo "built $IMAGE ($(docker image inspect "$IMAGE" --format '{{.Size}}') bytes)"

echo "=== 2. smoke test: does a scratch container even start? ==="
docker run --rm "${CAPS[@]}" "$IMAGE" 2>&1 | head -2 || true

rm -rf "$OUT"; mkdir -p "$OUT"

# Time a container the way the organizer does: the daemon's own window, which excludes
# create/inspect/remove but includes the runtime's start-up.
time_container() {
    local name="t3val-$RANDOM-$RANDOM"
    docker run --name "$name" "${CAPS[@]}" "$@" >/dev/null 2>&1 || true
    local t
    t="$(docker inspect -f '{{.State.StartedAt}} {{.State.FinishedAt}}' "$name" 2>/dev/null || echo "")"
    docker rm -f "$name" >/dev/null 2>&1 || true
    [ -z "$t" ] && { echo "nan"; return; }
    python3 - "$t" <<'PY'
import sys, datetime as dt
def ns(s):
    s = s.replace("Z", "+00:00"); head, _, rest = s.partition(".")
    frac_ns, tz = 0, ""
    if rest:
        frac, sign, tz = rest.partition("+"); frac_ns = int((frac + "000000000")[:9]); tz = sign + tz
    return int(dt.datetime.fromisoformat(head + (tz or "+00:00")).timestamp()) * 10**9 + frac_ns
a, b = sys.argv[1].split()
print(f"{(ns(b) - ns(a)) / 1e9:.6f}")
PY
}

echo "=== 3. run every unit ==="
printf '%-40s %10s %12s %14s\n' "unit" "events" "wall_s" "events/sec"
printf '%s\n' "--------------------------------------------------------------------------------"
total_eps=0; n=0
for u in "$REPO"/units/*/; do
    slug="$(basename "$u")"
    mkdir -p "$OUT/$slug"
    if [ -f "$u/scenario.json" ]; then
        secs="$(time_container -v "$(cd "$u" && pwd):/input:ro" -v "$OUT/$slug:/output" \
                "$IMAGE" simulate --config /input/scenario.json --out /output/trace.parquet)"
    else
        secs="$(time_container -v "$(cd "$u" && pwd):/input:ro" -v "$OUT/$slug:/output" \
                "$IMAGE" simulate-batch --batch-dir /input/scenarios --out-dir /output)"
    fi
    ev="$(python3 -c "
import json,glob,sys
fs=glob.glob('$OUT/$slug/events.json')+glob.glob('$OUT/$slug/*/events.json')
print(sum(json.load(open(f))['n_events'] for f in fs) if fs else 0)")"
    eps="$(python3 -c "print(f'{$ev/$secs:,.0f}' if $secs>0 else '-')" 2>/dev/null || echo '-')"
    printf '%-40s %10s %12s %14s\n' "$slug" "$ev" "$secs" "$eps"
    total_eps="$(python3 -c "print($total_eps + ($ev/$secs if $secs>0 else 0))")"
    n=$((n+1))
done
echo
python3 -c "print(f'arithmetic mean over {$n} units: {$total_eps/$n:,.0f} events/sec (the ranking statistic)')"

echo
echo "=== 4. grade with the gate ==="
( cd "$REPO" && python check_candidate.py --candidate "$OUT" 2>/dev/null \
    || python ../track3-work/check_candidate.py --candidate "$OUT" ) | tail -5
