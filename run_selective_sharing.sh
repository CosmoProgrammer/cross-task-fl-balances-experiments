#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Server-side follow-ups from the gradient-conflict keystone (env `yuv_fl`):
#   PART A (cheap, ~20 min): AdamW-DELTA conflict diagnostic — settles whether the
#     ~15x forecasting>anomaly RAW-gradient magnitude imbalance survives the real
#     local AdamW updates (the actual FedAvg pseudo-gradient). Run this FIRST.
#   PART B (multi-hour FL training): SELECTIVE backbone-sharing variants that test
#     the encoder-conflict finding (encoder cos<0, decoder cos>0):
#       B1  share decoders+gates, encoders TASK-PRIVATE   (remove the conflicting
#           module from cross-task sharing -> should help if encoder sharing hurts)
#       B2  share encoders ONLY, decoders+gates private    (the contrast: share only
#           the conflicting module -> should hurt)
#     Anchors already exist: dual = share-all, single_task = share-none.
#
# Set DIAG_ONLY=1 to run only PART A (get the magnitude answer before spending GPU
# hours on training). RESUMABLE: skips any step whose output JSON exists.
# AUTO commit+push of plain-git artifacts (never *.pt). Server push loop is done,
# so this is the sole pusher -> plain push fast-forwards.
#
# Knobs: PY (default "conda run -n yuv_fl python"), COHORT (50), SEEDS (42),
#        EVAL_EVERY (5), DIAG_ONLY (unset), NOPUSH=1, NOGIT=1.
# Usage:  bash run_selective_sharing.sh
#         DIAG_ONLY=1 bash run_selective_sharing.sh
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-conda run -n yuv_fl python}
COHORT=${COHORT:-50}
SEEDS=${SEEDS:-"42"}
EVAL_EVERY=${EVAL_EVERY:-5}
CT() { [[ "$COHORT" == "50" ]] && echo "" || echo "_c$COHORT"; }   # cohort tag
ST() { [[ "$1" == "42" ]] && echo "" || echo "_s$1"; }            # seed tag

run_step () {  # $1=output-to-skip-on  $2=label  $3=cmd
    local out="$1" label="$2" cmd="$3"
    if [[ -f "$out" ]]; then echo "--- SKIP (exists): $out"; return 0; fi
    echo "=== [$(date '+%F %T')] RUN: $label"
    if eval "$cmd"; then echo "=== done: $label"; else echo "!! FAILED: $label (continuing)"; fi
}

commit_push () {  # $1=message
    [[ -n "${NOGIT:-}" ]] && return 0
    git add results/*.json results/*.npz logs/ 2>/dev/null
    if git diff --cached --quiet; then echo "--- git: nothing new"; return 0; fi
    git commit -m "$1" >/dev/null 2>&1 && echo "--- git: committed ($1)" || { echo "!! commit failed"; return 0; }
    [[ -n "${NOPUSH:-}" ]] && { echo "--- git: NOPUSH"; return 0; }
    git push >/dev/null 2>&1 && echo "--- git: pushed" || echo "!! push failed (pull+retry)"
}

echo "########## SELECTIVE SHARING | cohort ${COHORT} | seeds ${SEEDS} | DIAG_ONLY=${DIAG_ONLY:-0} ##########"

# ── PART A: AdamW-delta magnitude/conflict diagnostic (cheap) ──
run_step "results/update_conflict_dual_c50_results.json" "delta-conflict dual c50" \
    "$PY analyze_update_conflict.py --tag dual_c50"
FC100="checkpoints/fed_dual_fedavg_c100_forecasting_model.pt"
AN100="checkpoints/fed_dual_fedavg_c100_anomaly_model.pt"
if [[ -f "$FC100" && -f "$AN100" ]]; then
    run_step "results/update_conflict_dual_c100_results.json" "delta-conflict dual c100" \
        "$PY analyze_update_conflict.py --cohort 100 --tag dual_c100 --fc-ckpt '$FC100' --an-ckpt '$AN100'"
fi
commit_push "selective-sharing PART A: AdamW-delta conflict diagnostic"

if [[ -n "${DIAG_ONLY:-}" ]]; then
    echo "########## DIAG_ONLY set — stopping before training variants ##########"; exit 0
fi

# ── PART B: selective-sharing FL training variants ──
# (each ~3-3.5h clean at c50, ~2x under contention)
for S in $SEEDS; do
    TAG="$(CT)$(ST "$S")"
    run_step "results/federated_dual_fedavg${TAG}_share-decoders-gates_results.json" \
        "B1 share decoders+gates (enc private) seed $S" \
        "$PY main.py --federated --mode dual --strategy fedavg --cohort $COHORT --seed $S \
            --eval-every $EVAL_EVERY --share-modules decoders,gates"
    commit_push "selective-sharing B1 (share decoders+gates) seed $S"

    run_step "results/federated_dual_fedavg${TAG}_share-encoders_results.json" \
        "B2 share encoders only seed $S" \
        "$PY main.py --federated --mode dual --strategy fedavg --cohort $COHORT --seed $S \
            --eval-every $EVAL_EVERY --share-modules encoders"
    commit_push "selective-sharing B2 (share encoders only) seed $S"
done

echo "########## SELECTIVE SHARING COMPLETE ##########"
echo "Laptop: git pull --ff-only; compare share-decoders-gates / share-encoders vs"
echo "        dual (share-all) + single_task (share-none) anchors."
