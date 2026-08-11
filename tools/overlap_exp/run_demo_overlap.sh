#!/usr/bin/env bash
# Run the GEMM || all-reduce overlap demo, end to end.
#
#   ./run_demo_overlap.sh                 # lock 1200 MHz, run, restore
#   ./run_demo_overlap.sh 900             # a different clock
#   ./run_demo_overlap.sh none            # don't touch clocks (no sudo needed)
#   ./run_demo_overlap.sh 1200 trtllm     # drive the collective through TensorRT-LLM
#
# Everything the demo needs beyond a stock PyTorch install is handled here:
#
#   * GCP's gIB NCCL shim breaks NCCL on A100 AND rejects NCCL_{MIN,MAX}_CTAS, which
#     is the knob that sets the collective's SM footprint. Unsetting NCCL_NET is not
#     enough -- /usr/local/gib must come off LD_LIBRARY_PATH.
#   * A locked clock is a request, not a fact, so the lock is verified by read-back
#     and the demo prints the clock it actually achieved.
#   * Clocks are restored on EVERY exit path, including Ctrl-C and any error. Leaving
#     a box pinned at a locked clock affects every other user of it.

set -euo pipefail

CLOCK="${1:-1200}"
BACKEND="${2:-torch}"
GPUS="0,1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- restore clocks AND persistence mode no matter how we leave ------------------
LOCKED=0
PM_WAS=""
cleanup() {
    if [ "$LOCKED" = "1" ]; then
        sudo nvidia-smi -i "$GPUS" -rgc >/dev/null 2>&1 || true
        echo "[clocks] restored"
    fi
    # persistence is a box-wide setting; put it back exactly as it was found
    if [ "$PM_WAS" = "Disabled" ]; then
        sudo nvidia-smi -i "$GPUS" -pm 0 >/dev/null 2>&1 || true
        echo "[clocks] persistence mode restored to Disabled"
    fi
}
trap cleanup EXIT INT TERM

# ---- prerequisites ---------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$NGPU" -ge 2 ] || { echo "need >= 2 GPUs, found $NGPU"; exit 1; }

PY="${PYTHON:-python3}"
[ "$BACKEND" = "trtllm" ] && PY="${TRTLLM_PYTHON:-$HOME/trtllm_venv/bin/python}"
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "python not found: $PY"; exit 1; }

"$PY" - <<'EOF' || { echo "missing python deps (need torch with CUDA, pynvml)"; exit 1; }
import sys
try:
    import torch, pynvml
    assert torch.cuda.is_available(), "torch sees no CUDA device"
except Exception as e:
    print(f"  {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
EOF

# ---- the gIB shim has to come off the loader path --------------------------------
export LD_LIBRARY_PATH="$(
    "$PY" -c "import os;print(':'.join(p for p in os.environ.get('LD_LIBRARY_PATH','').split(':') if p and 'gib' not in p))"
)"
unset NCCL_NET || true

# ---- lock the clock, and verify it took ------------------------------------------
if [ "$CLOCK" != "none" ]; then
    if sudo -n true 2>/dev/null || sudo true; then
        # WITHOUT PERSISTENCE MODE THE LOCK SILENTLY DOES NOT HOLD. With no client
        # holding the device the driver tears down GPU state and drops the locked
        # clock; -lgc still returns "All done." and the run then executes at the
        # boost clock. Observed here after a reboot cleared persistence: requested
        # 1200 MHz, ran at 1410.
        PM_WAS=$(nvidia-smi -i 0 --query-gpu=persistence_mode --format=csv,noheader)
        [ "$PM_WAS" = "Enabled" ] || sudo nvidia-smi -i "$GPUS" -pm 1 >/dev/null
        sudo nvidia-smi -i "$GPUS" -lgc "$CLOCK,$CLOCK" >/dev/null
        LOCKED=1
        sleep 1
        GOT=$(nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits)
        echo "[clocks] persistence ${PM_WAS} -> Enabled; requested ${CLOCK} MHz, "\
             "idle read-back ${GOT} MHz"
        # An idle GPU may sit below the lock; the number that matters is the clock
        # the demo samples DURING the workload, which it prints in its header.
    else
        echo "[clocks] no sudo -- running unlocked; the demo will report the achieved clock"
    fi
fi

echo "[run] $PY demo_overlap.py --backend $BACKEND"
echo
"$PY" demo_overlap.py --backend "$BACKEND"

echo
if [ -f trace_overlap.json ]; then
    echo "[trace] $HERE/trace_overlap.json  ($(du -h trace_overlap.json | cut -f1))"
    echo "        open it at https://ui.perfetto.dev  -- each CUDA stream is its own"
    echo "        row; overlapping bars on different rows are the overlap."
fi
