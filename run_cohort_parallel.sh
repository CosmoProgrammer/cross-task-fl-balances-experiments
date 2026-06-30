#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Process-parallel cohort sweep (run on the GPU server, env `yuv_fl`).
#
# Same trimmed grid + resumable skip-if-exists semantics as run_cohort_sweep.sh,
# but runs the 4 conditions as TWO WAVES of TWO independent single-GPU PROCESSES
# (one pinned to each GPU via CUDA_VISIBLE_DEVICES) instead of 4 sequential runs.
# ~2x total sweep throughput with ZERO model-code risk: separate processes, full
# isolation, each forced to single-GPU mode (one visible device => device_count==1
# even with config.max_federated_gpus=2). This is the SAFE speed path; the
# count-balanced 2-GPU split (intra-condition) is the other one and needs the
# per-group timing measurement first.
#
# Conditions write DISTINCT mode-tagged files, so concurrent runs never collide.
# Each condition logs to logs/parallel_<mode>_c<N>.log (interleaved stdout would
# be unreadable). Skips any condition whose tagged result JSON already exists.
#
# Usage (from repo root):
#   bash run_cohort_parallel.sh 200                  # GPUs 0,1
#   GPUS="0,1" bash run_cohort_parallel.sh 200
#   PYTHON="conda run -n yuv_fl python" bash run_cohort_parallel.sh 200
#
# CAUTION: assumes BOTH GPUs are free to you. On the shared box, if the other
# user holds a GPU, the pinned process there just contends (slower, not a crash).
# ─────────────────────────────────────────────────────────────────────────────
set -u  # NOT -e: one failed condition must not kill the sweep

cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
RESULTS_DIR=results
LOG_DIR=logs
mkdir -p "$LOG_DIR"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-0,1}"

if [[ $# -eq 0 ]]; then
    echo "usage: bash run_cohort_parallel.sh <cohort> [cohort ...]   (e.g. 200)"
    exit 2
fi

# Launch one condition on a given GPU IF its result JSON is missing.
# Echoes the launched PID on fd (via global LAUNCHED_PID); empty if skipped.
launch_if_needed () {  # $1=gpu  $2=out_json  $3=label  $4..=command
    local gpu="$1" out="$2" label="$3"; shift 3
    LAUNCHED_PID=""
    if [[ -f "$RESULTS_DIR/$out" ]]; then
        echo "--- SKIP (exists): $out"
        return 0
    fi
    local logf="$LOG_DIR/parallel_${label}.log"
    echo "=== [$(date '+%F %T')] START: $label on GPU $gpu  (-> $logf)"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$logf" 2>&1 &
    LAUNCHED_PID=$!
}

# Wait on a (pid label) pair list, reporting each exit code.
wait_pair () {  # args: pid1 label1 pid2 label2  (pid may be empty = skipped)
    local p1="$1" l1="$2" p2="$3" l2="$4"
    for pair in "$p1|$l1" "$p2|$l2"; do
        local pid="${pair%%|*}" lab="${pair##*|}"
        [[ -z "$pid" ]] && continue
        wait "$pid"; local rc=$?
        echo "=== [$(date '+%F %T')] END:   $lab  exit=$rc"
    done
}

for COHORT in "$@"; do
    if [[ "$COHORT" == "50" ]]; then
        echo "!! Refusing cohort 50: anchor is done and uses UNTAGGED outputs."
        continue
    fi
    TAG="_c${COHORT}"
    echo ""
    echo "########################## COHORT ${COHORT} (parallel, GPUs ${GPU_A},${GPU_B}) ##########################"

    # 1. Preprocess (sequential — must finish before any training reads the data)
    A_META="data/ashrae/processed_c${COHORT}/split_metadata.json"
    L_META="data/lead/processed_c${COHORT}/split_metadata.json"
    if [[ -f "$A_META" && -f "$L_META" ]]; then
        echo "--- SKIP preprocess (exists): processed_c${COHORT}"
    else
        echo "=== [$(date '+%F %T')] preprocess c${COHORT}"
        $PY preprocess.py --cohort "$COHORT" \
            || { echo "!! preprocess failed for c${COHORT}; skipping"; continue; }
    fi

    # 2. Wave 1: dual ‖ single_task
    launch_if_needed "$GPU_A" "federated_dual_fedavg${TAG}_results.json" "dual_c${COHORT}" \
        $PY main.py --federated --mode dual --strategy fedavg --cohort "$COHORT"
    P_DUAL=$LAUNCHED_PID
    launch_if_needed "$GPU_B" "federated_single_task_fedavg${TAG}_results.json" "single_task_c${COHORT}" \
        $PY main.py --federated --mode single_task --strategy fedavg --cohort "$COHORT"
    P_ST=$LAUNCHED_PID
    wait_pair "$P_DUAL" "dual_c${COHORT}" "$P_ST" "single_task_c${COHORT}"

    # 3. Wave 2: local_only ‖ centralized
    launch_if_needed "$GPU_A" "federated_local_only_fedavg${TAG}_results.json" "local_only_c${COHORT}" \
        $PY main.py --federated --mode local_only --strategy fedavg --cohort "$COHORT"
    P_LO=$LAUNCHED_PID
    launch_if_needed "$GPU_B" "centralized${TAG}_results.json" "centralized_c${COHORT}" \
        $PY main.py --centralized --cohort "$COHORT"
    P_CEN=$LAUNCHED_PID
    wait_pair "$P_LO" "local_only_c${COHORT}" "$P_CEN" "centralized_c${COHORT}"

    echo "########################## COHORT ${COHORT} DONE ##########################"
done

echo ""
echo "ALL REQUESTED COHORTS COMPLETE: $*"
