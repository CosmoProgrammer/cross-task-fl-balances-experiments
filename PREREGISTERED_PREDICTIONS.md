# PRE-REGISTERED PREDICTIONS — the when-to-share criterion vs the imputation grid

**Status: PRE-REGISTRATION. The git commit timestamp of this file is the commitment.** It is
committed **before any imputation task exists in code and before any imputation training run.** The
FC–AD anchor value below is locked as of this commit; the imputation predictions are stated here and
their outcomes appended *after* the runs, honestly, whichever way they go (Track 2 in
`THINGS_TO_DO.md`).

**What is being tested.** Theorem 3 (`theory/NOTES_theorem3.md`, MC-verified) gives a debiased,
estimable **share-or-split criterion** with an oracle guarantee. This file freezes that criterion,
computes it on the one task pair we already have data for (FC–AD, from committed dumps), and
pre-commits its **predicted verdict for the imputation pairs** — so the imputation grid is an
*out-of-sample* test of the criterion, not a post-hoc fit.

---

## 1. The frozen criterion (Theorem 3, target (a))

For a task pair {1,2} with `n` clients per task, from the per-client backbone deltas each client
produces in **one probe round** at the shared dual backbone:

- `S_t` := task-t **mean backbone delta** (the aggregate `d_fc` / `d_an` that FedAvg averages).
- `τ̂²` := **pooled within-task scatter** (unbiased): `τ̂² = [Σ_t Σ_{i∈t}‖d_i − μ_t‖²] / [Σ_t(n_t−1)]`.
- `γ̂² := τ̂²/n` — per-task estimation error of the mean.
- **`δ*² := 2γ̂²`** — the when-to-share threshold.
- **`T̂ := ‖S_1 − S_2‖² − 4γ̂²`** — the debiased regret statistic, `E[T̂] = δ² − 2γ² = 4·(R_dual − R_single)`.

**Decision rule:** **share (dual) iff `T̂ < 0`; else split (single_task).** Equivalently, share iff
`‖S_1−S_2‖² < 4τ̂²/n`. The `−4γ̂²` debiasing removes the `δ*²=2γ²` inflation of `‖S_1−S_2‖²` (each
mean contributes a `γ²`); it is load-bearing *near* the threshold (it is what makes the §6i
per-module null come out exactly single), though for FC–AD the sign is already decided without it.

Oracle guarantee (why we trust one probe round): `Regret = P(wrong)·|T|/4 ≤ std(T̂)/4`, and
`std(T̂) ∝ 1/√(d·R)` shrinks in the backbone dimension `d≈8.3e5` and probe replicates `R` — so the
verdict is sharply estimable (Theorem 3 §5–6).

**Reproducible instantiation:** `compute_affinity_criterion.py` (committed) reconstructs every inner
product exactly from the dumped per-client delta norms + full pairwise cosine matrix (their Gram
matrix) — no per-client delta vectors needed. `conda run -n crosstask python compute_affinity_criterion.py`.

---

## 2. ANCHOR (locked): FC–AD is a SPLIT, and the criterion agrees with the measurement

Computed from the **committed** dumps `results/update_conflict_dual_c{50,100}.npz` (+ `_results.json`),
which are the AdamW per-client backbone deltas at the shared dual backbone (seed 42):

| pair | cohort | n | `‖S_1−S_2‖²` | `τ̂²` | `δ*²=2γ̂²` | **`T̂`** | ratio `‖·‖²/δ*²` | verdict |
|---|---|---|---|---|---|---|---|---|
| FC–AD | c50 | 35 | 95.84 | 191.5 | 10.94 | **+73.96** | 8.8× | **SPLIT (single_task)** |
| FC–AD | c100 | 85 | 89.12 | 196.3 | 4.62 | **+79.88** | 19.3× | **SPLIT (single_task)** |

(Uniform-mean vs weighted-aggregate `S_t` agree to <0.4%; weighted shown — the real-data map.)

**This is a genuine consistency win.** The criterion, computed purely from delta geometry, says FC–AD
**should not share (dual hurts)** — which is exactly the measured hierarchy `dual < single_task`
(EXPERIMENT_RESULTS §6e / the c100 grid). The margin is large (8.8×–19.3× above threshold), so the
measured `dual < single` is **far from a knife-edge**, not seed noise. And `T̂` grows c50→c100
(8.8×→19.3×): the split deepens as `n` rises (δ* shrinks ∝1/√n), matching that dual gets *worse*
relative to single with more clients. Debiasing is not sign-critical here (`‖S_1−S_2‖²=95.8 ≫ 4γ̂²=21.9`).

⚠ The FC–AD anchor is only a **one-sided** test: the criterion has so far only been checked where it
says SPLIT. The imputation grid tests the **SHARE** side — the harder, more falsifiable direction.

---

## 3. PREDICTIONS for the imputation pairs (to be tested)

Design (`THINGS_TO_DO.md` §3.2, lean option A): a third task **IMP = ASHRAE mask-and-reconstruct
imputation** (`out_len=128`, all windows), on a **disjoint** group of 35 ASHRAE buildings. This
**decouples data domain from objective type**:

| task | data domain | objective | `out_len` |
|---|---|---|---|
| FC (forecast) | ASHRAE | predict next 24 | 24 |
| AD (anomaly) | LEAD | reconstruct (clean windows) | 128 |
| IMP (impute) | ASHRAE | reconstruct (masked positions) | 128 |

**Central hypothesis: update-direction affinity is driven by OBJECTIVE type (loss geometry:
reconstruct vs forecast), not by data domain.** The backbone update direction is shaped by what the
loss asks the network to do, not by the input distribution. Predicted signs of `T̂` (each computed
from that pair's *own* probe dump, once IMP exists):

| pair | data | objective | **predicted `T̂`** | **predicted verdict** | rationale |
|---|---|---|---|---|---|
| **AD–IMP** | LEAD vs ASHRAE (different) | both reconstruct (**same**) | **`T̂ < 0`** | **SHARE (dual wins)** | same loss geometry ⇒ small δ; the sharp new-data test |
| **FC–IMP** | both ASHRAE (**same**) | forecast vs reconstruct (different) | `T̂ > 0`, smaller margin than FC–AD | SPLIT (or near threshold) | different objective ⇒ separated δ, but shared domain may shrink τ̂²/δ vs FC–AD |
| **FC–AD** | different | different | `T̂ > 0` (**measured: +74/+80**) | SPLIT | the locked anchor |
| FC–AD–IMP (triple) | mixed | mixed | dominated by FC separation | single-ish; dual pulled toward the FC–AD split | the FC leg carries the largest δ |

**The ordering pre-committed** (sharing benefit `R_single − R_dual`, most-positive = dual helps most):
```
   AD–IMP   >   FC–IMP   >   FC–AD
   (share)      (~split)     (split, measured)
```
If the "data domain dominates" alternative were true instead, the ordering would flip to
`FC–IMP > AD–IMP` (both-ASHRAE pairs share) — so the grid **discriminates the two hypotheses**.

---

## 4. Falsification (pre-committed)

The criterion is validated iff **`sign(T̂)` predicts `sign(R_single − R_dual)`** for each pair, i.e.
`T̂<0` ⇒ dual beats single, `T̂>0` ⇒ single beats dual (beyond multi-seed noise; use the crux-pair
error bars). Specifically, the criterion is **FALSIFIED** if any of:

1. **AD–IMP: `T̂ < 0` but dual(AD+IMP) does NOT beat single** — a false SHARE. This is the primary,
   sharpest test (the first predicted share).
2. **AD–IMP: `T̂ > 0`** at all — would refute the "same-objective ⇒ share" hypothesis outright
   (before any run), and mean objective type does *not* drive affinity here.
3. **FC–IMP: sign(`T̂`) disagrees with the observed best topology.**
4. The pre-committed **ordering** `AD–IMP > FC–IMP > FC–AD` is violated in the measured
   `R_single − R_dual`.

A clean pass = the criterion, frozen before the data existed, predicts the correct share/split for
all three new pairs and the ordering holds.

---

## 5. Scope & honest caveats

- **One-shot probe, single seed.** The FC–AD dumps are one probe round at seed 42. Theorem 3's bound
  is for a one-shot probe (clean independence). Multi-seed the crux pair (AD–IMP) at c50 (43–47).
- **AdamW deltas, not SGD.** The statistics are AdamW per-client deltas (the object FedAvg averages),
  not the SGD gradients the theory idealizes — the delta-vs-gradient lens (EXPERIMENT_RESULTS §6g) is
  *why* we measure at the update level. Theorem 2 (`NOTES_theorem2.md` §6) states the AdamW≠SGD gap.
- **`τ̂²` is large relative to `‖S_1−S_2‖²`** (191 vs 96): within-task client scatter dominates a
  single probe round. This is fine — the debiasing accounts for it exactly — but it means the *raw*
  separation is modest; the split verdict rests on `‖S_1−S_2‖²` still exceeding `4γ̂² = 4τ̂²/n` after
  the `1/n` shrink, which it does by a wide margin.
- **New pairs need their own dumps.** AD–IMP / FC–IMP `T̂` require an `analyze_update_conflict.py`
  generalized to arbitrary task pairs / the IMP model (Track 3.5). Same script, same formula.
- **Per-module rule available.** The blockwise `T̂_b` (Theorem 3 §7) reproduced the §6i selective-
  sharing null exactly; if a pair's global `T̂` is near zero, report the per-module breakdown too.

---

## 6. OUTCOMES (append after the imputation grid runs — leave blank until then)

| pair | predicted `T̂` sign | predicted verdict | measured `T̂` | measured best topology | criterion correct? |
|---|---|---|---|---|---|
| AD–IMP | `< 0` | SHARE | _pending_ | _pending_ | _pending_ |
| FC–IMP | `> 0` (small) | SPLIT / near | _pending_ | _pending_ | _pending_ |
| FC–AD | `> 0` | SPLIT | +73.96 (c50) / +79.88 (c100) | single_task (measured) | ✅ (anchor) |
| triple | — | single-ish | _pending_ | _pending_ | _pending_ |

_Do not edit §§1–5 after commit except to correct a transcription error (note any such edit). §6 is
the only section filled post-hoc._
