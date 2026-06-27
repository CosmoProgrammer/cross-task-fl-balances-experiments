#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Weekend sweep (GPU server, env `yuv_fl`): multi-seed c50 + a c200 saturation point.
#
# WHY: (1) every conclusion so far is single-seed -> the dual-vs-single_task gap
# (~0.016 AUC) is not separable from seed variance. Cohort scaling SATURATES, so
# c50 is statistically valid AND ~half the cost of c100 -> buy error bars at c50.
# (2) The saturation curve only has TWO x-points (c50, c100); a third is wanted.
# Federated c200 (~22-25h/condition contended) won't finish this window, so we add
# the cheap, clean substitute: CENTRALIZED c200 (~9h, pooled upper bound, no FL
# confound) as the third saturation point.
#
# WHAT (CRUX-FIRST so the headline lands even if contention eats the tail):
#   Phase 1 (headline):  dual ‖ single_task at every seed              [c50, ~20h]
#   Phase 2 (support):   centralized c200 (3rd sat point) + c50 context
#                        (local_only ‖ centralized at every seed)      [best-effort]
# Both phases run as process-parallel PAIRS across the two GPUs (one condition
# pinned per GPU via CUDA_VISIBLE_DEVICES; one visible device => single-GPU mode
# even with config.max_federated_gpus=2). ZERO model-code risk (isolated procs).
# Seeds != 42 tag every output `_s{seed}` (config.cohort_tag) so runs + the
# committed c50 anchor coexist; nothing is clobbered.
#
# AUTO-COMMIT: after each pair, narrowly stage results/*.json + logs/ (PLAIN git --
# NEVER checkpoints/*.pt, which are LFS and broken on the server) and commit+push,
# so the laptop pulls fresh numbers as they land. Push failure is non-fatal (the
# next commit catches up). Only ONE committer (sequential pairs) => no git races.
#
# RESUMABLE: skips any condition whose tagged result JSON already exists.
#
# Usage (from repo root, inside tmux/nohup so it survives disconnect):
#   PYTHON="conda run -n yuv_fl python" bash run_multiseed_weekend.sh
#   SEEDS="43 44 45 46" PYTHON="conda run -n yuv_fl python" bash run_multiseed_weekend.sh
#   RUN_C200=0 ... bash run_multiseed_weekend.sh        # skip the c200 saturation point
#   GPUS="0,1" NOPUSH=1 ... bash run_multiseed_weekend.sh   # commit but don't push
#
# Env knobs: SEEDS (default "43 44 45 46 47"), GPUS (default "0,1"),
#            PYTHON (default "python"), RUN_C200 (default 1), COHORT (default 50),
#            EVAL_EVERY (default 5; federated per-round eval cadence, final round
#            always evals so reported metrics are unchanged), NOPUSH=1 (commit, no
#            push), NOGIT=1 (no git at all).
# ─────────────────────────────────────────────────────────────────────────────
set -u  # NOT -e: one failed condition must not kill the sweep

cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
COHORT=${COHORT:-50}
SEEDS=${SEEDS:-"43 44 45 46 47"}
RUN_C200=${RUN_C200:-1}
EVAL_EVERY=${EVAL_EVERY:-5}   # per-round TEST eval cadence for federated runs; final
                             # round always evals (reported metrics unchanged), so K=5
                             # just sparsifies history and saves ~9% wall. Set 1 for
                             # dense per-round curves.
RESULTS_DIR=results
LOG_DIR=logs
mkdir -p "$LOG_DIR"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-0,1}"

ctag() { [[ "$1" == "50" ]] && echo "" || echo "_c$1"; }   # cohort-only filename tag

# ---- data presence check for the main cohort (multi-seed reuses ONE fixed split)
if [[ "$COHORT" == "50" ]]; then
    A_META="data/ashrae/processed/split_metadata.json"
    L_META="data/lead/processed/split_metadata.json"
else
    A_META="data/ashrae/processed_c${COHORT}/split_metadata.json"
    L_META="data/lead/processed_c${COHORT}/split_metadata.json"
fi
if [[ ! -f "$A_META" || ! -f "$L_META" ]]; then
    echo "!! Missing processed data for cohort ${COHORT} ($A_META / $L_META)."
    echo "!! Run: $PY preprocess.py --cohort ${COHORT}   (multi-seed reuses one fixed split)"
    exit 3
fi

# ---- kick off c200 preprocess in the BACKGROUND now (CPU work, ready by Phase 2)
PREP_PID=""
if [[ "$RUN_C200" == "1" && ! -f data/lead/processed_c200/split_metadata.json ]]; then
    echo "=== [$(date '+%F %T')] preprocess c200 in background (-> $LOG_DIR/preprocess_c200.log)"
    bash -c "$PY preprocess.py --cohort 200" > "$LOG_DIR/preprocess_c200.log" 2>&1 &
    PREP_PID=$!
fi

# ---- launch one condition (cmd STRING) on a GPU if its result JSON is missing ----
launch_if_needed () {  # $1=gpu  $2=out_json  $3=label  $4=cmd_string
    local gpu="$1" out="$2" label="$3" cmd="$4"
    LAUNCHED_PID=""
    if [[ -f "$RESULTS_DIR/$out" ]]; then echo "--- SKIP (exists): $out"; return 0; fi
    local logf="$LOG_DIR/weekend_${label}.log"
    echo "=== [$(date '+%F %T')] START: $label on GPU $gpu  (-> $logf)"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" > "$logf" 2>&1 &
    LAUNCHED_PID=$!
}

wait_one () { local pid="$1" lab="$2"; [[ -z "$pid" ]] && return 0; \
              wait "$pid"; echo "=== [$(date '+%F %T')] END:   $lab  exit=$?"; }

# ---- narrow commit+push of plain-git artifacts (never .pt) -----------------------
commit_push () {  # $1 = message
    [[ -n "${NOGIT:-}" ]] && return 0
    git add "$RESULTS_DIR"/*.json "$LOG_DIR" 2>/dev/null
    if git diff --cached --quiet; then echo "--- git: nothing new to commit"; return 0; fi
    git commit -m "$1" >/dev/null 2>&1 && echo "--- git: committed ($1)" \
        || { echo "!! git commit failed"; return 0; }
    [[ -n "${NOPUSH:-}" ]] && { echo "--- git: NOPUSH set, skipping push"; return 0; }
    if git push >/dev/null 2>&1; then echo "--- git: pushed"
    else echo "!! git push failed (commits queued; next push catches up)"; fi
}

# ---- run parallel arrays Q_OUT / Q_LABEL / Q_CMD two-at-a-time across both GPUs ---
run_queue () {
    local n=${#Q_OUT[@]} i=0
    while (( i < n )); do
        launch_if_needed "$GPU_A" "${Q_OUT[$i]}" "${Q_LABEL[$i]}" "${Q_CMD[$i]}"
        local pa=$LAUNCHED_PID la="${Q_LABEL[$i]}" pb="" lb=""
        if (( i + 1 < n )); then
            launch_if_needed "$GPU_B" "${Q_OUT[$((i+1))]}" "${Q_LABEL[$((i+1))]}" "${Q_CMD[$((i+1))]}"
            pb=$LAUNCHED_PID; lb="${Q_LABEL[$((i+1))]}"
        fi
        wait_one "$pa" "$la"; wait_one "$pb" "$lb"
        commit_push "weekend sweep: ${la}${lb:+ + }${lb} done"
        i=$((i + 2))
    done
}

CT="$(ctag "$COHORT")"
echo "########## WEEKEND SWEEP | cohort ${COHORT} | seeds: ${SEEDS} | GPUs ${GPU_A},${GPU_B} | RUN_C200=${RUN_C200} ##########"

# ===== PHASE 1 — CRUX: dual ‖ single_task, all seeds (the headline) ==============
Q_OUT=(); Q_LABEL=(); Q_CMD=()
for S in $SEEDS; do
    ST="${CT}_s${S}"
    Q_OUT+=("federated_dual_fedavg${ST}_results.json");        Q_LABEL+=("dual${ST}")
    Q_CMD+=("$PY main.py --federated --mode dual --strategy fedavg --cohort $COHORT --seed $S --eval-every ${EVAL_EVERY}")
    Q_OUT+=("federated_single_task_fedavg${ST}_results.json"); Q_LABEL+=("single_task${ST}")
    Q_CMD+=("$PY main.py --federated --mode single_task --strategy fedavg --cohort $COHORT --seed $S --eval-every ${EVAL_EVERY}")
done
echo ""; echo "===== PHASE 1 (crux): dual ‖ single_task, seeds ${SEEDS} ====="
run_queue

# ===== PHASE 2 — SUPPORT: centralized c200 (3rd sat point) + c50 context =========
Q_OUT=(); Q_LABEL=(); Q_CMD=()
if [[ "$RUN_C200" == "1" ]]; then
    [[ -n "$PREP_PID" ]] && { echo "=== [$(date '+%F %T')] waiting for c200 preprocess"; wait "$PREP_PID"; }
    if [[ -f data/lead/processed_c200/split_metadata.json ]]; then
        Q_OUT+=("centralized_c200_results.json"); Q_LABEL+=("centralized_c200")
        Q_CMD+=("$PY main.py --centralized --cohort 200")
    else
        echo "!! c200 preprocess did not produce data; skipping centralized c200"
    fi
fi
for S in $SEEDS; do
    ST="${CT}_s${S}"
    Q_OUT+=("federated_local_only_fedavg${ST}_results.json"); Q_LABEL+=("local_only${ST}")
    Q_CMD+=("$PY main.py --federated --mode local_only --strategy fedavg --cohort $COHORT --seed $S --eval-every ${EVAL_EVERY}")
    Q_OUT+=("centralized${ST}_results.json"); Q_LABEL+=("centralized${ST}")
    Q_CMD+=("$PY main.py --centralized --cohort $COHORT --seed $S")
done
echo ""; echo "===== PHASE 2 (support): centralized c200 + c50 context (local_only ‖ centralized) ====="
run_queue

echo ""; echo "########## WEEKEND SWEEP COMPLETE (cohort ${COHORT}, seeds ${SEEDS}) ##########"
