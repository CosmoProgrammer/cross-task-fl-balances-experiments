#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION CRITERION GRID (GPU server, env `yuv_fl`) — THINGS_TO_DO §3.2.
#
# WHY: the third task (imputation = mask-and-reconstruct on a DISJOINT ASHRAE
# cohort) is the pre-registered OUT-OF-SAMPLE test of the Theorem-3 share-or-split
# criterion. Because IMP uses ASHRAE data but a reconstruct objective, it
# decouples data-domain from objective-type. Pre-registered predictions
# (PREREGISTERED_PREDICTIONS.md, committed BEFORE any run here):
#   AD-IMP  (diff data, SAME objective)  -> T_hat<0 -> SHARE (dual beats single)  [PRIMARY test]
#   FC-IMP  (SAME data, diff objective)  -> ~split / near threshold
#   FC-AD   (anchor)                     -> split (measured dual<single)
#   ordering of sharing benefit:  AD-IMP > FC-IMP > FC-AD
#
# WHAT (c50, seed 42 first): the generic --tasks trainer path. Each pair is run
# in BOTH dual (shares backbone) and single_task (no sharing) so the per-task
# metric gap = the measured R_single - R_dual for the criterion table.
# Ordered by pre-registration priority: AD-IMP first, then IMP baselines, then
# FC-IMP, then the FC+AD+IMP triple.
#
# OUTPUT (tag `_tasks-<tokens>`, so nothing clobbers the 2-task FC+AD anchors):
#   results/federated_dual_fedavg_tasks-an-imp_results.json          (AD-IMP share)
#   results/federated_single_task_fedavg_tasks-an-imp_results.json   (AD-IMP no-share)
#   results/federated_single_task_fedavg_tasks-imp_results.json      (IMP baseline)
#   results/federated_local_only_fedavg_tasks-imp_results.json       (IMP no-exchange)
#   results/federated_dual_fedavg_tasks-fc-imp_results.json          (FC-IMP share)
#   results/federated_single_task_fedavg_tasks-fc-imp_results.json   (FC-IMP no-share)
#   results/federated_dual_fedavg_tasks-fc-an-imp_results.json       (triple)
#
# ⚠ PRE-REGISTRATION GUARD: refuses to run until PREREGISTERED_PREDICTIONS.md is
# COMMITTED to git (the commit timestamp IS the pre-registration — non-negotiable
# ordering). Override only for a dry test with PREREG_OK=1.
#
# TIMING: c50 pair (70 clients) ~20min/round clean -> ~3.5h/run; IMP-only (35
# clients) ~10min/round; triple (105 clients) ~30min/round. x1.5-2.3 under
# contention. Serial single-GPU by default (server usually contended). RESUMABLE:
# skips any condition whose result JSON already exists.
#
# Usage (from repo root, inside tmux/nohup):
#   PYTHON="conda run -n yuv_fl python" bash run_imputation.sh            # GPU 0, serial
#   GPUS="0,1" PYTHON="conda run -n yuv_fl python" bash run_imputation.sh # parallel pairs
#   SEEDS="42 43 44 45 46 47" ... bash run_imputation.sh                  # multi-seed crux
#   NOPUSH=1 ... / NOGIT=1 ...                                            # git control
#
# Env knobs: SEEDS (default "42"), GPUS (default "0"), PYTHON (default python),
#            COHORT (default 50), ROUNDS (default 10), EVAL_EVERY (default 1),
#            ONLY (optional space list of labels to restrict to), PREREG_OK=1
#            (bypass the pre-registration guard — dry tests only),
#            NOPUSH=1, NOGIT=1.
# ─────────────────────────────────────────────────────────────────────────────
set -u  # NOT -e: one failed condition must not kill the sweep

cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
COHORT=${COHORT:-50}
SEEDS=${SEEDS:-"42"}
ROUNDS=${ROUNDS:-10}
EVAL_EVERY=${EVAL_EVERY:-1}
RESULTS_DIR=results
LOG_DIR=logs
mkdir -p "$LOG_DIR"
IFS=',' read -r GPU_A GPU_B <<< "${GPUS:-0}"

ctag()   { [[ "$1" == "50" ]] && echo "" || echo "_c$1"; }
seedtag(){ [[ "$1" == "42" ]] && echo "" || echo "_s$1"; }

# ---- PRE-REGISTRATION GUARD (non-negotiable ordering) --------------------------
if [[ -z "${PREREG_OK:-}" ]]; then
    if ! git ls-files --error-unmatch PREREGISTERED_PREDICTIONS.md >/dev/null 2>&1; then
        echo "!! PREREGISTERED_PREDICTIONS.md is NOT committed to git."
        echo "!! The commit timestamp IS the pre-registration — it MUST land BEFORE"
        echo "!! the first imputation training run. Commit it (and"
        echo "!! compute_affinity_criterion.py), then re-run. Bypass (dry test only):"
        echo "!!   PREREG_OK=1 bash run_imputation.sh"
        exit 4
    fi
fi

# ---- data presence: base FC/AD + the imputation group --------------------------
if [[ "$COHORT" == "50" ]]; then
    A_META="data/ashrae/processed/split_metadata.json"
    L_META="data/lead/processed/split_metadata.json"
    I_META="data/imputation/processed/split_metadata.json"
else
    A_META="data/ashrae/processed_c${COHORT}/split_metadata.json"
    L_META="data/lead/processed_c${COHORT}/split_metadata.json"
    I_META="data/imputation/processed_c${COHORT}/split_metadata.json"
fi
if [[ ! -f "$A_META" || ! -f "$L_META" ]]; then
    echo "!! Missing base processed data for cohort ${COHORT}."
    echo "!! Run: $PY preprocess.py --cohort ${COHORT}"
    exit 3
fi
if [[ ! -f "$I_META" ]]; then
    echo "=== Imputation data missing; building it (disjoint ASHRAE, cohort ${COHORT}) ..."
    $PY preprocess.py --imputation --skip-base --cohort "$COHORT" \
        > "$LOG_DIR/preprocess_imputation_c${COHORT}.log" 2>&1
    if [[ ! -f "$I_META" ]]; then
        echo "!! Imputation preprocessing failed — see the log. Aborting."
        exit 3
    fi
    echo "=== Imputation data built: $I_META"
fi

launch_if_needed () {  # $1=gpu  $2=out_json  $3=label  $4=cmd_string
    local gpu="$1" out="$2" label="$3" cmd="$4"
    LAUNCHED_PID=""
    if [[ -n "${ONLY:-}" ]] && [[ " $ONLY " != *" $label "* ]]; then
        echo "--- SKIP (not in ONLY): $label"; return 0; fi
    if [[ -f "$RESULTS_DIR/$out" ]]; then echo "--- SKIP (exists): $out"; return 0; fi
    local logf="$LOG_DIR/imputation_${label}.log"
    echo "=== [$(date '+%F %T')] START: $label on GPU $gpu  (-> $logf)"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" > "$logf" 2>&1 &
    LAUNCHED_PID=$!
}

wait_one () { local pid="$1" lab="$2"; [[ -z "$pid" ]] && return 0; \
              wait "$pid"; echo "=== [$(date '+%F %T')] END:   $lab  exit=$?"; }

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

run_queue () {
    local n=${#Q_OUT[@]} i=0
    if [[ -z "$GPU_B" || "$GPU_B" == "$GPU_A" ]]; then      # SERIAL single GPU
        while (( i < n )); do
            launch_if_needed "$GPU_A" "${Q_OUT[$i]}" "${Q_LABEL[$i]}" "${Q_CMD[$i]}"
            wait_one "$LAUNCHED_PID" "${Q_LABEL[$i]}"
            commit_push "imputation grid: ${Q_LABEL[$i]} done"
            i=$((i + 1))
        done
        return 0
    fi
    while (( i < n )); do                                    # PARALLEL pairs
        launch_if_needed "$GPU_A" "${Q_OUT[$i]}" "${Q_LABEL[$i]}" "${Q_CMD[$i]}"
        local pa=$LAUNCHED_PID la="${Q_LABEL[$i]}" pb="" lb=""
        if (( i + 1 < n )); then
            launch_if_needed "$GPU_B" "${Q_OUT[$((i+1))]}" "${Q_LABEL[$((i+1))]}" "${Q_CMD[$((i+1))]}"
            pb=$LAUNCHED_PID; lb="${Q_LABEL[$((i+1))]}"
        fi
        wait_one "$pa" "$la"; wait_one "$pb" "$lb"
        commit_push "imputation grid: ${la}${lb:+ + }${lb} done"
        i=$((i + 2))
    done
}

# base command for one condition
cmd_for () {  # $1=mode  $2=tasks  $3=seed
    echo "$PY main.py --tasks $2 --mode $1 --strategy fedavg --cohort $COHORT --seed $3 --rounds $ROUNDS --eval-every ${EVAL_EVERY}"
}

CT="$(ctag "$COHORT")"
echo "########## IMPUTATION GRID | cohort ${COHORT} | seeds: ${SEEDS} | GPUs ${GPU_A},${GPU_B:-} | rounds ${ROUNDS} ##########"

# Ordered by pre-registration priority. (label = filename mode_tasks tokens.)
Q_OUT=(); Q_LABEL=(); Q_CMD=()
add () {  # $1=mode  $2=tasks(comma)  $3=seed
    local mode="$1" tasks="$2" S="$3"
    local tok="${tasks//,/-}"
    local ST="${CT}$(seedtag "$S")"
    Q_OUT+=("federated_${mode}_fedavg_tasks-${tok}${ST}_results.json")
    Q_LABEL+=("${mode}_tasks-${tok}${ST}")
    Q_CMD+=("$(cmd_for "$mode" "$tasks" "$S")")
}

for S in $SEEDS; do
    add dual        an,imp    "$S"   # PRIMARY: AD-IMP share
    add single_task an,imp    "$S"   # AD-IMP no-share
    add single_task imp       "$S"   # IMP baseline (within-group FedAvg)
    add local_only  imp       "$S"   # IMP baseline (no exchange)
    add dual        fc,imp    "$S"   # FC-IMP share
    add single_task fc,imp    "$S"   # FC-IMP no-share
    add dual        fc,an,imp "$S"   # triple (secondary)
done
run_queue

echo ""; echo "########## IMPUTATION GRID COMPLETE (cohort ${COHORT}, seeds ${SEEDS}) ##########"
echo "Next: build the criterion table (compute_affinity_criterion on each pair's"
echo "dual dump) vs the measured per-task R_single - R_dual; append outcomes to"
echo "PREREGISTERED_PREDICTIONS.md §6 and EXPERIMENT_RESULTS.md."
