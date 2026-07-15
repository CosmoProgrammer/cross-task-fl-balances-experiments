"""
Laptop-side logic validation of the SCAFFOLD implementation in
`trainers/multitask_fed_trainer.py`.  NO real data, NO GPU — synthetic clients,
tiny model, CPU, eager path.  Exercises the REAL trainer code paths.

Validates the four checks from THINGS_TO_DO.md 3.1:
  (a) control variates c_i, c initialise to zero
  (b) the g - c_i + c correction is applied to the gradient each local step
      BEFORE the optimizer step
  (c) the c_i / c update rules match the SCAFFOLD paper (which option?)
  (d) NULL TEST: with c_i = c = 0 throughout, scaffold == fedavg byte-identically.

Plus a liveness check (scaffold must actually DIVERGE from fedavg once the
control variates become non-zero) so the null test can't pass by the code being
silently dead.

Run:  conda run -n crosstask python validate_scaffold.py
"""
import copy
import logging

import numpy as np
import torch

from configs.config import ExperimentConfig
from main import build_forecasting_model, build_anomaly_model, build_aux_loss
from trainers.multitask_fed_trainer import MultiTaskFederatedTrainer

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SEED = 1234


def make_config(num_rounds):
    """Real architecture, shrunk hidden dims (dims are irrelevant to the
    SCAFFOLD logic) so the eager CPU path runs in seconds."""
    c = ExperimentConfig()
    c.hid_len, c.hid_chn, c.hid_pch, c.hid_pred = 32, 32, 16, 32
    c.d_ssm, c.state_size = 16, 8
    c.local_epochs = 2
    c.num_rounds = num_rounds
    c.max_federated_gpus = 1
    c.use_mamba_kernel = False
    c.eval_every = 10_000          # skip per-round eval noise; not needed here
    return c


def make_client_loaders(n_clients, task, seed):
    """Synthetic per-client loaders. shuffle=False for determinism."""
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for i in range(n_clients):
        x = torch.randn(8, 128, 1, generator=g)
        if task == "fc":
            y = torch.randn(8, 24, 1, generator=g)
        else:                      # anomaly: labels unused during training
            y = torch.zeros(8)
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        loaders[f"{task}{i}"] = {"train": dl, "n_samples": 8}
    return loaders


def make_test_loader(task, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 128, 1, generator=g)
    y = torch.randn(4, 24, 1, generator=g) if task == "fc" else torch.zeros(4)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)


def run_one(strategy, fc_init, an_init, num_rounds):
    """Run `num_rounds` of federated training from a FIXED init and return the
    two global state_dicts (CPU tensors)."""
    cfg = make_config(num_rounds)
    cfg.aggregation_mode = "dual"
    cfg.fl_strategy = strategy

    fc = copy.deepcopy(fc_init)
    an = copy.deepcopy(an_init)

    # Seed IMMEDIATELY before train() so the stochastic streams that both runs
    # share (dropout, the anomaly mask = torch.rand_like) are byte-identical.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    trainer = MultiTaskFederatedTrainer(
        fc, an, torch.device("cpu"), cfg,
        aux_loss_fn=build_aux_loss(cfg),
        logger=logging.getLogger("t"),
    )
    fc_loaders = make_client_loaders(2, "fc", seed=10)
    an_loaders = make_client_loaders(2, "an", seed=20)
    trainer.train(fc_loaders, an_loaders,
                  make_test_loader("fc", 30), make_test_loader("an", 40))

    controls = None
    if strategy in ("scaffold", "scaffold_c1"):
        controls = {
            "global_fc": trainer.global_control_fc,
            "client_fc": trainer.client_controls_fc,
        }
    return (
        {k: v.detach().cpu().clone() for k, v in fc.state_dict().items()},
        {k: v.detach().cpu().clone() for k, v in an.state_dict().items()},
        controls,
    )


def max_abs_diff(sd_a, sd_b):
    m = 0.0
    for k in sd_a:
        d = (sd_a[k].float() - sd_b[k].float()).abs().max().item()
        m = max(m, d)
    return m


def main():
    print("=" * 70)
    print("SCAFFOLD logic validation (synthetic, CPU)")
    print("=" * 70)

    # Fixed init shared by every run.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fc_init = build_forecasting_model(make_config(1))
    an_init = build_anomaly_model(make_config(1))

    # ---- NULL TEST: 1 round, fedavg vs scaffold ----------------------------
    fa_fc, fa_an, _ = run_one("fedavg", fc_init, an_init, num_rounds=1)
    sc_fc, sc_an, sc_ctrl = run_one("scaffold", fc_init, an_init, num_rounds=1)

    d_fc = max_abs_diff(fa_fc, sc_fc)
    d_an = max_abs_diff(fa_an, sc_an)
    print(f"\n[NULL TEST]  round-1 scaffold vs fedavg  (expect 0.0)")
    print(f"    forecasting max|Δ| = {d_fc:.3e}")
    print(f"    anomaly     max|Δ| = {d_an:.3e}")
    null_ok = (d_fc == 0.0 and d_an == 0.0)
    print(f"    => {'PASS' if null_ok else 'FAIL'}  "
          f"(controls start at 0 -> correction is a no-op in round 1)")

    # ---- (a) controls initialised to zero ----------------------------------
    #     Inspected via a fresh trainer's _init_scaffold below.
    cfg = make_config(1); cfg.aggregation_mode = "dual"; cfg.fl_strategy = "scaffold"
    t = MultiTaskFederatedTrainer(copy.deepcopy(fc_init), copy.deepcopy(an_init),
                                  torch.device("cpu"), cfg, logger=logging.getLogger("t"))
    t._init_scaffold()
    gzero = all(v.abs().max().item() == 0.0 for v in t.global_control_fc.values())
    empty = (len(t.client_controls_fc) == 0 and len(t.client_controls_an) == 0)
    print(f"\n[INIT]  global control all-zero = {gzero}; "
          f"client controls empty (lazily zero) = {empty}")

    # ---- controls become NON-zero after the update -------------------------
    gmax = max(v.abs().max().item() for v in sc_ctrl["global_fc"].values())
    cmax = max(
        max(v.abs().max().item() for v in cdict.values())
        for cdict in sc_ctrl["client_fc"].values()
    )
    print(f"\n[UPDATE]  after round 1: max|global_control_fc| = {gmax:.3e}, "
          f"max|client_control_fc| = {cmax:.3e}")
    print(f"    client update LIVE (non-zero): "
          f"{'PASS' if cmax > 0 else 'FAIL'}")

    # ---- SCAFFOLD INVARIANT:  c == mean_i(c_i)  under full participation ----
    # Correct SCAFFOLD keeps the global control equal to the mean of the client
    # controls at all times (c=0=mean holds at init; each round's update
    # preserves it). Check it after round 1.
    client_dicts = list(sc_ctrl["client_fc"].values())
    inv_err = 0.0
    for name in sc_ctrl["global_fc"]:
        mean_ci = sum(cd[name] for cd in client_dicts) / len(client_dicts)
        inv_err = max(inv_err, (sc_ctrl["global_fc"][name] - mean_ci).abs().max().item())
    print(f"    INVARIANT  max|c - mean(c_i)| = {inv_err:.3e}  (expect ~0)")
    inv_ok = inv_err < 1e-6
    print(f"    => global control tracks client mean: "
          f"{'PASS' if inv_ok else 'FAIL  <-- global control update is BROKEN'}")

    # ---- LIVENESS: 2 rounds must DIVERGE (controls now active) --------------
    fa2_fc, fa2_an, _ = run_one("fedavg", fc_init, an_init, num_rounds=2)
    sc2_fc, sc2_an, _ = run_one("scaffold", fc_init, an_init, num_rounds=2)
    d2 = max(max_abs_diff(fa2_fc, sc2_fc), max_abs_diff(fa2_an, sc2_an))
    print(f"\n[LIVENESS]  round-2 scaffold vs fedavg  (expect > 0)")
    print(f"    max|Δ| = {d2:.3e}  => "
          f"{'PASS (scaffold is live)' if d2 > 0 else 'FAIL (scaffold is dead)'}")

    # ====================================================================
    # OPTION I (strategy "scaffold_c1"): control = mean grad at global model
    # ====================================================================
    print("\n" + "=" * 70)
    print("OPTION I  (scaffold_c1)  — control recomputed at the global model")
    print("=" * 70)

    # NULL TEST: round-1 controls are 0 -> correction is a no-op -> byte-identical
    # to fedavg (same as Option II; only the END-of-round control UPDATE differs).
    c1_fc, c1_an, c1_ctrl = run_one("scaffold_c1", fc_init, an_init, num_rounds=1)
    d1_fc = max_abs_diff(fa_fc, c1_fc)
    d1_an = max_abs_diff(fa_an, c1_an)
    print(f"\n[NULL TEST]  round-1 scaffold_c1 vs fedavg  (expect 0.0)")
    print(f"    forecasting max|Δ| = {d1_fc:.3e}")
    print(f"    anomaly     max|Δ| = {d1_an:.3e}")
    c1_null_ok = (d1_fc == 0.0 and d1_an == 0.0)
    print(f"    => {'PASS' if c1_null_ok else 'FAIL'}")

    # Control update LIVE and FINITE (mean grad at global should be non-zero).
    c1_cmax = max(
        max(v.abs().max().item() for v in cdict.values())
        for cdict in c1_ctrl["client_fc"].values()
    )
    c1_finite = all(
        torch.isfinite(v).all().item()
        for cdict in c1_ctrl["client_fc"].values() for v in cdict.values()
    )
    print(f"\n[UPDATE]  after round 1: max|client_control_fc| = {c1_cmax:.3e}  "
          f"(mean grad at global)")
    print(f"    control non-zero = {c1_cmax > 0}; all finite = {c1_finite}")

    # INVARIANT c == mean(c_i) must still hold under Option I's global update.
    c1_client_dicts = list(c1_ctrl["client_fc"].values())
    c1_inv_err = 0.0
    for name in c1_ctrl["global_fc"]:
        mean_ci = sum(cd[name] for cd in c1_client_dicts) / len(c1_client_dicts)
        c1_inv_err = max(c1_inv_err,
                         (c1_ctrl["global_fc"][name] - mean_ci).abs().max().item())
    c1_inv_ok = c1_inv_err < 1e-6
    print(f"    INVARIANT  max|c - mean(c_i)| = {c1_inv_err:.3e}  => "
          f"{'PASS' if c1_inv_ok else 'FAIL <-- BUG'}")

    # LIVENESS: round-2 must diverge from fedavg (controls now active), and it
    # must ALSO differ from Option II (different control -> different trajectory).
    c1_2_fc, c1_2_an, _ = run_one("scaffold_c1", fc_init, an_init, num_rounds=2)
    c1_d2 = max(max_abs_diff(fa2_fc, c1_2_fc), max_abs_diff(fa2_an, c1_2_an))
    c1_vs_c2 = max(max_abs_diff(sc2_fc, c1_2_fc), max_abs_diff(sc2_an, c1_2_an))
    print(f"\n[LIVENESS]  round-2 scaffold_c1 vs fedavg    max|Δ| = {c1_d2:.3e}  => "
          f"{'PASS' if c1_d2 > 0 else 'FAIL (dead)'}")
    print(f"[DISTINCT]  round-2 scaffold_c1 vs Option II  max|Δ| = {c1_vs_c2:.3e}  => "
          f"{'PASS (Option I != Option II)' if c1_vs_c2 > 0 else 'FAIL (identical)'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  [II ] null test (correction injection correct) : {'PASS' if null_ok else 'FAIL'}")
    print(f"  [II ] client control update live               : {'PASS' if cmax > 0 else 'FAIL'}")
    print(f"  [II ] scaffold diverges from fedavg (liveness) : {'PASS' if d2 > 0 else 'FAIL'}")
    print(f"  [II ] global control == mean(client controls)  : {'PASS' if inv_ok else 'FAIL <-- BUG'}")
    print(f"  [I  ] null test (round-1 == fedavg)            : {'PASS' if c1_null_ok else 'FAIL'}")
    print(f"  [I  ] control live + finite                    : {'PASS' if (c1_cmax > 0 and c1_finite) else 'FAIL'}")
    print(f"  [I  ] global control == mean(client controls)  : {'PASS' if c1_inv_ok else 'FAIL <-- BUG'}")
    print(f"  [I  ] diverges from fedavg (liveness)          : {'PASS' if c1_d2 > 0 else 'FAIL'}")
    print(f"  [I  ] distinct from Option II                  : {'PASS' if c1_vs_c2 > 0 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
