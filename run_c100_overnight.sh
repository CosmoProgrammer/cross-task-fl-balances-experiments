#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight cohort-100 run. Runs the PROPOSED method first (the must-have), then
# the two ablations, then centralized. Each condition runs INDEPENDENTLY — a
# failure/crash in one does NOT block the rest (important for an unbabysat run).
#
# Recommended: ONE idle GPU (reliable, predictable ~48 min/round, ~8h/condition):
#     CUDA_VISIBLE_DEVICES=0 bash run_c100_overnight.sh      # use the idle GPU's index
#
# To gamble on 2 GPUs instead (NOT recommended — the one 2-GPU measurement showed
# ~no speedup, 2993s vs a ~2900s 1-GPU prediction, and the threaded path is
# unverified past 1 round): expose both AND set config.max_federated_gpus=2:
#     CUDA_VISIBLE_DEVICES=0,1 bash run_c100_overnight.sh    # + edit configs/config.py
#
# Detach from tmux with Ctrl-b then d; the server keeps running it.
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}

run () {  # $1 = label, rest = command; times it, never aborts the others on failure
    local label="$1"; shift
    local t0; t0=$(date +%s)
    echo "=== [$(date '+%F %T')] START: $label"
    "$@"; local rc=$?
    echo "=== [$(date '+%F %T')] END:   $label  exit=$rc  ($(( ($(date +%s) - t0) / 60 ))m)"
}

echo "GPUs visible to this run: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Cohort 100 | rounds = config default (10)"

run "dual/fedavg (PROPOSED)" $PY main.py --federated --cohort 100 --mode dual        --strategy fedavg
run "single_task/fedavg"     $PY main.py --federated --cohort 100 --mode single_task --strategy fedavg
run "local_only/fedavg"      $PY main.py --federated --cohort 100 --mode local_only  --strategy fedavg
run "centralized"            $PY main.py --centralized --cohort 100

echo "=== ALL CONDITIONS ATTEMPTED: $(date '+%F %T') ==="
