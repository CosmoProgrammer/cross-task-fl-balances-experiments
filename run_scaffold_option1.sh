#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD Option I probe (GPU server, env `yuv_fl`) — THINGS_TO_DO §3.1 next step.
#
# WHY: SCAFFOLD Option II (strategy `scaffold`) HURT both tasks on c50/seed42
# (AN AUC dual 0.798->0.730, single ~0.812->0.724; FC R2 dual 0.937->0.854,
# single ~0.944->0.819; forecast test loss RISES r5->r10). That is the predicted
# Option-II x AdamW scale-miscalibration signature: Option II derives the control
# from (x_global - x_trained)/(K*eta), a sum of RMS-normalized AdamW steps, not a
# gradient. The prescribed fallback (07-14) is Option I, NOT SGD: recompute the
# control as the mean gradient at the GLOBAL model each round — optimizer-agnostic,
# keeps AdamW so the comparison to the fedavg anchors stays clean and a null becomes
# trustworthy. New code: `--strategy scaffold_c1` (trainer scaffold_option=1).
#
# WHAT: c50, seed 42, dual ‖ single_task with `--strategy scaffold_c1`, num_rounds 10.
# Runs as a process-parallel PAIR across the two GPUs (one condition pinned per GPU
# via CUDA_VISIBLE_DEVICES; one visible device => single-GPU mode even with
# config.max_federated_gpus=2). ZERO shared-model-state risk (isolated processes).
#
# COMPARE (done off the JSONs afterwards, not here) against, at c50/seed42:
#   fedavg anchor:  dual AUC 0.798 / R2 0.937 ; single ~0.812 / ~0.944
#   scaffold (II):  dual AUC 0.730 / R2 0.854 ; single 0.724 / 0.819
#   => Option I "helps" if it recovers toward (or past) the fedavg anchor and the
#      forecast test-loss curve stops rising r5->r10. Only if STILL null -> SGD.
#
# OUTPUT (c50/seed42 => cohort_tag ""):
#   results/federated_dual_scaffold_c1_results.json
#   results/federated_single_task_scaffold_c1_results.json
# Distinct from Option II (`_scaffold_`) and fedavg (`_fedavg_`) — nothing clobbered.
#
# TIMING DISCIPLINE (standing rule): watch the round-1 wall-clock in the logs; a
# clean c50 round is ~20 min (Option I adds ~20% for the extra grad-at-global pass:
# 1 pass on top of local_epochs=5). If round 1 is pathological (contention), Ctrl-C
# and retry later. A full 10-round run is ~3.5-4h clean, x1.5-2.3 under contention;
# the pair runs concurrently so the two finish together.
#
# AUTO-COMMIT: after the pair, narrowly stage results/*.json + logs/ (PLAIN git —
# NEVER checkpoints/*.pt, which are LFS and broken on the server) and commit+push.
# RESUMABLE: skips any condition whose result JSON already exists.
#
# Usage (from repo root, inside tmux/nohup so it survives disconnect):
#   PYTHON="conda run -n yuv_fl python" bash run_scaffold_option1.sh
#   SEEDS="42 43 44 45 46 47" PYTHON="conda run -n yuv_fl python" bash run_scaffold_option1.sh
#   GPUS="0,1" NOPUSH=1 ... bash run_scaffold_option1.sh      # commit but don't push
#   NOGIT=1 ... bash run_scaffold_option1.sh                  # no git at all
#
# Env knobs: SEEDS (default "42"), GPUS (default "0,1"), PYTHON (default python),
#            COHORT (default 50), ROUNDS (default 10), EVAL_EVERY (default 1 —
#            DENSE per-round eval so the r5->r10 forecast-loss curve is visible;
#            final round always evals so reported metrics are unaffected regardless),
#            NOPUSH=1 (commit, no push), NOGIT=1 (no git at all).
# ─────────────────────────────────────────────────────────────────────────────
set -u  # NOT -e: one failed condition must not kill the sweep

cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
COHORT=${COHORT:-50}
SEEDS=${SEEDS:-"42"}
ROUNDS=${ROUNDS:-10}
EVAL_EVERY=${EVAL_EVERY:-1}   # dense per-round eval (the destabilization signature
                             # is per-round); final round always evals so REPORTED
                             # metrics are unchanged regardless of this value.
RESULTS_DIR=results
LOG_DIR=logs
mkdir -p "$LOG_DIR"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-0,1}"

ctag() { [[ "$1" == "50" ]] && echo "" || echo "_c$1"; }   # cohort-only filename tag
seedtag() { [[ "$1" == "42" ]] && echo "" || echo "_s$1"; } # seed!=42 => _s{seed}

# ---- data presence check (multi-seed reuses ONE fixed split at the preprocess seed)
if [[ "$COHORT" == "50" ]]; then
    A_META="data/ashrae/processed/split_metadata.json"
    L_META="data/lead/processed/split_metadata.json"
else
    A_META="data/ashrae/processed_c${COHORT}/split_metadata.json"
    L_META="data/lead/processed_c${COHORT}/split_metadata.json"
fi
if [[ ! -f "$A_META" || ! -f "$L_META" ]]; then
    echo "!! Missing processed data for cohort ${COHORT} ($A_META / $L_META)."
    echo "!! Run: $PY preprocess.py --cohort ${COHORT}"
    exit 3
fi

# ---- launch one condition (cmd STRING) on a GPU if its result JSON is missing ----
launch_if_needed () {  # $1=gpu  $2=out_json  $3=label  $4=cmd_string
    local gpu="$1" out="$2" label="$3" cmd="$4"
    LAUNCHED_PID=""
    if [[ -f "$RESULTS_DIR/$out" ]]; then echo "--- SKIP (exists): $out"; return 0; fi
    local logf="$LOG_DIR/scaffold_c1_${label}.log"
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
        commit_push "scaffold Option I (scaffold_c1): ${la}${lb:+ + }${lb} done"
        i=$((i + 2))
    done
}

CT="$(ctag "$COHORT")"
echo "########## SCAFFOLD OPTION I | cohort ${COHORT} | seeds: ${SEEDS} | GPUs ${GPU_A},${GPU_B} | rounds ${ROUNDS} ##########"

# dual ‖ single_task with strategy scaffold_c1, per seed
Q_OUT=(); Q_LABEL=(); Q_CMD=()
for S in $SEEDS; do
    ST="${CT}$(seedtag "$S")"
    Q_OUT+=("federated_dual_scaffold_c1${ST}_results.json");        Q_LABEL+=("dual_scaffold_c1${ST}")
    Q_CMD+=("$PY main.py --federated --mode dual --strategy scaffold_c1 --cohort $COHORT --seed $S --rounds $ROUNDS --eval-every ${EVAL_EVERY}")
    Q_OUT+=("federated_single_task_scaffold_c1${ST}_results.json"); Q_LABEL+=("single_task_scaffold_c1${ST}")
    Q_CMD+=("$PY main.py --federated --mode single_task --strategy scaffold_c1 --cohort $COHORT --seed $S --rounds $ROUNDS --eval-every ${EVAL_EVERY}")
done
run_queue

echo ""; echo "########## SCAFFOLD OPTION I COMPLETE (cohort ${COHORT}, seeds ${SEEDS}) ##########"
echo "Next: interpret results/federated_{dual,single_task}_scaffold_c1${CT}_results.json"
echo "vs the fedavg anchors + Option II scaffold JSONs (see EXPERIMENT_RESULTS.md §10)."
