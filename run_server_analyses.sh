#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Server-side post-sweep analyses (GPU box, env `yuv_fl`). All EVAL-ONLY / cheap
# (no retraining): per-seed anomaly score dumps for the cross-seed §6d re-check,
# plus gradient-conflict robustness runs for the pivot keystone. Run AFTER the
# multi-seed sweep, while the per-seed checkpoints (LFS, server-only) are present.
#
# WHAT:
#   1. eval_from_checkpoint --dump-scores for dual + single_task, seeds 44-47
#      (the seeds with BOTH arms) + dual s43 -> anomaly_scores_<mode>_s<seed>.npz
#      (repo root, matching the existing anomaly_scores_c100*.npz convention;
#      plain-git, pushed -> the laptop runs analyze_recon_decomposition.py on them).
#   2. analyze_gradient_conflict.py at the dual c50 backbone (idempotent re-run),
#      a --random-init round-0 baseline ("tasks conflict before training"), and
#      the c100 dual backbone if those checkpoints exist.
#
# RESUMABLE: skips any step whose output already exists. Safe to re-run.
# AUTO-COMMIT+PUSH (plain git, never *.pt): the new JSON/npz so the laptop pulls
# them. Server's weekend push loop is DONE, so this is the first pusher -> a plain
# push fast-forwards (no pull needed). Knobs: NOPUSH=1 (commit, no push),
# NOGIT=1 (no git at all), PY (default "conda run -n yuv_fl python"),
# SEEDS (default "44 45 46 47"), COHORT (default 50).
#
# Usage:  bash run_server_analyses.sh
#         NOPUSH=1 bash run_server_analyses.sh
#
# NOTE: checkpoint names are inferred as fed_<mode>_fedavg_s<seed>_<task>_model.pt
# (and ..._c100_... for c100). If `ls checkpoints/ | grep s4` shows different
# names, edit CKPT() below. Missing checkpoints are SKIPPED with a warning, not fatal.
# ─────────────────────────────────────────────────────────────────────────────
set -u  # NOT -e: one missing checkpoint must not kill the rest
cd "$(dirname "$0")" || exit 1

PY=${PY:-conda run -n yuv_fl python}
COHORT=${COHORT:-50}
SEEDS=${SEEDS:-"44 45 46 47"}

CKPT() { echo "checkpoints/fed_$1_fedavg$2_$3_model.pt"; }  # $1=mode $2=tag $3=task

run_step () {  # $1=output-to-check-for-skip  $2=label  $3=cmd
    local out="$1" label="$2" cmd="$3"
    if [[ -f "$out" ]]; then echo "--- SKIP (exists): $out"; return 0; fi
    echo "=== [$(date '+%F %T')] RUN: $label"
    if eval "$cmd"; then echo "=== done: $label"
    else echo "!! FAILED: $label (continuing)"; fi
}

dump () {  # $1=mode $2=seed  -- per-seed anomaly score dump
    local m="$1" s="$2" tag="_s${s}"
    local fc an; fc="$(CKPT "$m" "$tag" forecasting)"; an="$(CKPT "$m" "$tag" anomaly)"
    if [[ ! -f "$fc" || ! -f "$an" ]]; then
        echo "--- SKIP (no checkpoint): $m s$s  ($fc / $an)"; return 0; fi
    run_step "anomaly_scores_${m}_s${s}.npz" "dump ${m} s${s}" \
        "$PY eval_from_checkpoint.py --cohort $COHORT --tag ${m}_s${s} \
            --fc-ckpt '$fc' --an-ckpt '$an' \
            --dump-scores anomaly_scores_${m}_s${s}.npz"
}

echo "########## SERVER ANALYSES | cohort ${COHORT} | seeds ${SEEDS} ##########"

# ── 1. per-seed score dumps (dual + single_task across the paired seeds) ──
for s in $SEEDS; do
    dump dual "$s"
    dump single_task "$s"
done
dump dual 43   # dual has s43 (single_task s43 is the missing/deferred backfill)

# ── 2. gradient-conflict keystone + robustness ──
run_step "results/grad_conflict_dual_c50_results.json" "grad-conflict dual c50" \
    "$PY analyze_gradient_conflict.py --tag dual_c50"

run_step "results/grad_conflict_random_init_results.json" "grad-conflict random-init (round 0)" \
    "$PY analyze_gradient_conflict.py --random-init --tag random_init"

FC100="$(CKPT dual _c100 forecasting)"; AN100="$(CKPT dual _c100 anomaly)"
if [[ -f "$FC100" && -f "$AN100" ]]; then
    run_step "results/grad_conflict_dual_c100_results.json" "grad-conflict dual c100" \
        "$PY analyze_gradient_conflict.py --cohort 100 --tag dual_c100 \
            --fc-ckpt '$FC100' --an-ckpt '$AN100'"
else
    echo "--- SKIP (no c100 dual checkpoint): grad-conflict dual c100"
fi

# ── 3. stage + commit + push the plain-git artifacts (NEVER *.pt) ──
if [[ -z "${NOGIT:-}" ]]; then
    git add anomaly_scores_*_s*.npz results/grad_conflict_*.json \
            results/grad_conflict_*.npz results/reeval_*_results.json 2>/dev/null
    if git diff --cached --quiet; then
        echo "--- git: nothing new to commit"
    else
        git commit -m "server analyses: per-seed score dumps + gradient-conflict keystone" \
            >/dev/null 2>&1 && echo "--- git: committed" || echo "!! git commit failed"
        if [[ -n "${NOPUSH:-}" ]]; then echo "--- git: NOPUSH set, not pushing"
        elif git push >/dev/null 2>&1; then echo "--- git: pushed"
        else echo "!! git push failed (pull on laptop side and retry)"; fi
    fi
fi

echo "########## SERVER ANALYSES COMPLETE ##########"
echo "Next: on the laptop -> git pull --ff-only, then I'll run the cross-seed"
echo "      decomposition + visualize the gradient-conflict npz."
