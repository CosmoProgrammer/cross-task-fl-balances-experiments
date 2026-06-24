#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cohort-scaling sweep driver (run on the GPU server, env `yuv_fl`).
#
# RESUMABLE: each step is skipped if its output already exists, so if the box
# shuts down mid-sweep you just re-launch with the same args and it picks up
# where it left off. Per-condition granularity (a condition is saved only when
# it finishes); a crash *inside* a condition restarts that condition — but the
# trainer also writes `*_latest.pt` every round as crash insurance.
#
# Per the agreed grid-trimming rule, each NEW cohort runs only the cross-task-
# scaling question: dual/fedavg (proposed) + single_task/fedavg + local_only/
# fedavg + centralized. The FULL grid (fedprox + the 5 SOTA baselines) stays at
# the anchor cohort 50, which is already done — this script refuses cohort 50.
#
# Usage (from repo root):
#   bash run_cohort_sweep.sh 100              # one cohort
#   bash run_cohort_sweep.sh 100 200          # the ladder
#   bash run_cohort_sweep.sh 100 200 400      # + 400 stretch (uses all of LEAD)
#   PYTHON="conda run -n yuv_fl python" bash run_cohort_sweep.sh 100
#
# NOTE: round time is ~linear in cohort size (200≈4x, 400≈8x a 50-round). To use
# both GPUs set config.max_federated_gpus=2 (and validate a 1-round run first) —
# that's a config edit, not a CLI flag.
# ─────────────────────────────────────────────────────────────────────────────
set -u  # deliberately NOT `set -e`: one failed condition must not kill the sweep

cd "$(dirname "$0")" || exit 1   # always run from repo root
PY=${PYTHON:-python}
RESULTS_DIR=results

if [[ $# -eq 0 ]]; then
    echo "usage: bash run_cohort_sweep.sh <cohort> [cohort ...]   (e.g. 100 200 400)"
    exit 2
fi

run_cmd () {  # $1 = label, rest = command; times it and reports exit code
    local label="$1"; shift
    local t0; t0=$(date +%s)
    echo "=== [$(date '+%F %T')] START: $label"
    "$@"; local rc=$?
    local t1; t1=$(date +%s)
    echo "=== [$(date '+%F %T')] END:   $label  exit=$rc  ($(( (t1 - t0) / 60 ))m $(( (t1 - t0) % 60 ))s)"
    return $rc
}

for COHORT in "$@"; do
    if [[ "$COHORT" == "50" ]]; then
        echo "!! Refusing cohort 50: the anchor is done and uses UNTAGGED outputs; "
        echo "!! re-running it here would clobber the committed 50-cohort results."
        continue
    fi
    TAG="_c${COHORT}"
    echo ""
    echo "########################## COHORT ${COHORT} (tag=${TAG}) ##########################"

    # 1. Preprocess -> data/{ashrae,lead}/processed_c${COHORT}/  (skip if present)
    A_META="data/ashrae/processed_c${COHORT}/split_metadata.json"
    L_META="data/lead/processed_c${COHORT}/split_metadata.json"
    if [[ -f "$A_META" && -f "$L_META" ]]; then
        echo "--- SKIP preprocess (exists): processed_c${COHORT}"
    else
        run_cmd "preprocess c${COHORT}" $PY preprocess.py --cohort "$COHORT" \
            || { echo "!! preprocess failed for c${COHORT}; skipping its runs"; continue; }
    fi

    # 2. Federated conditions (trimmed grid: proposed + 2 ablations)
    for MODE in dual single_task local_only; do
        OUT="federated_${MODE}_fedavg${TAG}_results.json"
        if [[ -f "$RESULTS_DIR/$OUT" ]]; then
            echo "--- SKIP (exists): $OUT"
        else
            run_cmd "fed ${MODE}/fedavg c${COHORT}" \
                $PY main.py --federated --mode "$MODE" --strategy fedavg --cohort "$COHORT"
        fi
    done

    # 3. Centralized upper bound
    OUT="centralized${TAG}_results.json"
    if [[ -f "$RESULTS_DIR/$OUT" ]]; then
        echo "--- SKIP (exists): $OUT"
    else
        run_cmd "centralized c${COHORT}" $PY main.py --centralized --cohort "$COHORT"
    fi

    echo "########################## COHORT ${COHORT} DONE ##########################"
done

echo ""
echo "ALL REQUESTED COHORTS COMPLETE: $*"
