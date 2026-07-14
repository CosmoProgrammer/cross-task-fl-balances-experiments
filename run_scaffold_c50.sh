#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD c50 runner (GPU server, env `yuv_fl`) — Naman's ask + Theorem 2's arm.
#
# CONTEXT: SCAFFOLD is fully implemented in multitask_fed_trainer.py but was never
# validated. Laptop logic validation (2026-07-06, validate_scaffold.py) PASSED after
# fixing one bug (global control was never updated on a client's first round -> the
# invariant c == mean(c_i) is now exact). AdamW decision = Option (i): feed the
# g - c_i + c correction to AdamW (Option II control update as-is), so this changes
# ONLY the strategy and stays directly comparable to the committed fedavg anchors
# (dual 0.798 AUC / single ~0.812-0.816). See THINGS_TO_DO.md 3.1.
#
# WHAT:
#   STEP 0 (smoke):  1 round, dual scaffold -> crash/timing/loss-sanity check only.
#                    Its 1-round result JSON is DELETED afterwards so it can't be
#                    mistaken for (or block) the real run.
#   STEP 1 (main):   scaffold x {dual, single_task}, full rounds, run as a
#                    process-parallel PAIR across both GPUs (one condition pinned
#                    per GPU via CUDA_VISIBLE_DEVICES => single-GPU mode each).
#
# INTERPRETATION (write your read BEFORE looking — THINGS_TO_DO 3.1):
#   lifts both equally -> drift orthogonal to sharing (footnote)
#   closes dual-single gap -> Theorem 2 headline; extend to seeds
#   hurts -> suspect Option-II x AdamW scale miscalibration; try Option I (NOT SGD)
#            before concluding (fallback ladder in THINGS_TO_DO 3.1).
#
# RESUMABLE: skips a condition whose result JSON already exists.
# AUTO-COMMIT: narrowly stages results/*.json + logs/ (PLAIN git, NEVER .pt/LFS).
#
# Usage (from repo root, inside tmux/nohup):
#   PYTHON="conda run -n yuv_fl python" bash run_scaffold_c50.sh
#   SMOKE_ONLY=1 PYTHON="conda run -n yuv_fl python" bash run_scaffold_c50.sh
#   GPUS="0,1" NOPUSH=1 PYTHON="conda run -n yuv_fl python" bash run_scaffold_c50.sh
#
# Env knobs: PYTHON (default python), GPUS (default 0,1), COHORT (default 50),
#            EVAL_EVERY (default 5), SMOKE_ONLY=1, NOPUSH=1, NOGIT=1.
# ─────────────────────────────────────────────────────────────────────────────
set -u  # NOT -e: a failed condition must not kill the script mid-way

cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
COHORT=${COHORT:-50}
EVAL_EVERY=${EVAL_EVERY:-5}
RESULTS_DIR=results
LOG_DIR=logs
mkdir -p "$LOG_DIR"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-0,1}"

ctag() { [[ "$1" == "50" ]] && echo "" || echo "_c$1"; }
CT="$(ctag "$COHORT")"

# ---- data presence check (scaffold reuses the same fixed c50 split) --------------
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

echo "########## SCAFFOLD c${COHORT} | GPUs ${GPU_A},${GPU_B} | eval_every ${EVAL_EVERY} ##########"

# ===== STEP 0 — SMOKE: 1 round, dual scaffold ====================================
SMOKE_JSON="$RESULTS_DIR/federated_dual_scaffold${CT}_results.json"
SMOKE_LOG="$LOG_DIR/scaffold_smoke${CT}.log"
echo ""; echo "===== STEP 0 (smoke): 1-round dual scaffold (-> $SMOKE_LOG) ====="
had_json_before=0; [[ -f "$SMOKE_JSON" ]] && had_json_before=1
CUDA_VISIBLE_DEVICES="$GPU_A" bash -c \
  "$PY main.py --federated --mode dual --strategy scaffold --cohort $COHORT --rounds 1 --eval-every 1" \
  > "$SMOKE_LOG" 2>&1
smoke_rc=$?
echo "--- smoke exit=$smoke_rc"
echo "--- smoke log tail:"; tail -n 15 "$SMOKE_LOG" | sed 's/^/      /'
# Remove the throwaway 1-round JSON so it can't be mistaken for / block the real run.
# (Only delete if WE created it this run — never nuke a pre-existing full result.)
if [[ "$had_json_before" == "0" && -f "$SMOKE_JSON" ]]; then
    rm -f "$SMOKE_JSON"; echo "--- removed throwaway smoke JSON ($SMOKE_JSON)"
fi
if [[ $smoke_rc -ne 0 ]]; then
    echo "!! smoke FAILED (exit $smoke_rc) — fix before the main runs. Stopping."; exit 4
fi
if [[ -n "${SMOKE_ONLY:-}" ]]; then echo "=== SMOKE_ONLY set — stopping after smoke."; exit 0; fi

# ===== STEP 1 — MAIN: scaffold x {dual, single_task}, full rounds ================
launch_if_needed () {  # $1=gpu $2=out_json $3=label $4=cmd
    local gpu="$1" out="$2" label="$3" cmd="$4"
    LAUNCHED_PID=""
    if [[ -f "$RESULTS_DIR/$out" ]]; then echo "--- SKIP (exists): $out"; return 0; fi
    local logf="$LOG_DIR/scaffold_${label}.log"
    echo "=== [$(date '+%F %T')] START: $label on GPU $gpu (-> $logf)"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" > "$logf" 2>&1 &
    LAUNCHED_PID=$!
}
wait_one () { local pid="$1" lab="$2"; [[ -z "$pid" ]] && return 0; \
              wait "$pid"; echo "=== [$(date '+%F %T')] END:   $lab exit=$?"; }

echo ""; echo "===== STEP 1 (main): dual ‖ single_task, scaffold, full rounds ====="
launch_if_needed "$GPU_A" "federated_dual_scaffold${CT}_results.json" "dual${CT}" \
  "$PY main.py --federated --mode dual --strategy scaffold --cohort $COHORT --eval-every ${EVAL_EVERY}"
PA=$LAUNCHED_PID
launch_if_needed "$GPU_B" "federated_single_task_scaffold${CT}_results.json" "single_task${CT}" \
  "$PY main.py --federated --mode single_task --strategy scaffold --cohort $COHORT --eval-every ${EVAL_EVERY}"
PB=$LAUNCHED_PID
wait_one "$PA" "dual${CT}"; wait_one "$PB" "single_task${CT}"
commit_push "scaffold c${COHORT}: dual + single_task (Option i / AdamW)"

echo ""; echo "########## SCAFFOLD c${COHORT} COMPLETE ##########"
echo "Compare vs fedavg anchors (dual 0.798 AUC / single ~0.812-0.816); then fill the"
echo "interpretation matrix in THINGS_TO_DO 3.1 and log numbers in EXPERIMENT_RESULTS.md."
