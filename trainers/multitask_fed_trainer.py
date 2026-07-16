"""
Multi-Task Federated Trainer with configurable aggregation and FL strategies.

Generic over a REGISTRY of tasks (a `_TaskSpec` list). Each task carries its own
global model, client loaders, training "kind" (forecast vs masked-reconstruct),
and (for SCAFFOLD) its own control variates. The backbone is shared across ALL
tasks' clients; each task's heads (+ any task-private backbone modules) are
averaged within that task's client group.

  Built-in tasks (see main.py): forecasting (ASHRAE), anomaly (LEAD),
  imputation (disjoint ASHRAE). Any subset can run via `--tasks`.

Backward compatibility: the legacy 2-task constructor
`MultiTaskFederatedTrainer(fc_model, an_model, ...)` + `train(fc_loaders,
an_loaders, fc_test, an_test)` is preserved and executes byte-identically to the
pre-refactor trainer (task order [forecasting, anomaly], same aggregation
summation order, same RNG consumption). The committed FC+AD anchors reproduce.

Aggregation modes:
  - "dual":        Backbone across ALL clients, heads within task (proposed)
  - "single_task": FedAvg ALL params within each task group independently
  - "local_only":  No aggregation fed back; clients train from own checkpoints

FL strategies:
  - "fedavg":      Standard Federated Averaging
  - "fedprox":     FedAvg + proximal term (Li et al., 2020)
  - "scaffold":    SCAFFOLD Option II (reuses training deltas)
  - "scaffold_c1": SCAFFOLD Option I (control = mean grad recomputed at global)

Architecture split (MambaMixer / SSD-Net):
  Backbone: patch_encoders, patch_decoders, cross_scale_gates
  Heads:    pred_heads, scale_router
"""
import copy
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from models.mamba_mixer import MambaMixer


# Parameter classification
BACKBONE_PREFIXES = ("patch_encoders.", "patch_decoders.", "cross_scale_gates.")


def _is_backbone(name):
    return any(name.startswith(p) for p in BACKBONE_PREFIXES)


class _TaskSpec:
    """One task in the federated registry.

    Args:
        name:      unique task id (e.g. "forecasting", "anomaly", "imputation")
        csv_key:   short prefix used for CSV columns / history keys / legacy
                   compat ("forecast", "anomaly", "imputation")
        model:     the task's global MambaMixer
        kind:      "forecast" (x,y -> MSE) | "reconstruct" (mask-and-reconstruct)
        client_loaders / test_loader / val_loader: assigned at train() time
    """

    def __init__(self, name, csv_key, model, kind,
                 client_loaders=None, test_loader=None, val_loader=None):
        assert kind in ("forecast", "reconstruct")
        self.name = name
        self.csv_key = csv_key
        self.model = model
        self.kind = kind
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        # runtime state (set during training)
        self.global_control = None      # SCAFFOLD global control (CPU dict)
        self.client_controls = None     # SCAFFOLD per-client controls
        self.persistent = None          # local_only per-client state_dicts
        self._round_models = None
        self._round_weights = None


class MultiTaskFederatedTrainer:
    """Federated trainer with configurable aggregation and strategies."""

    def __init__(self, forecasting_model, anomaly_model, device, config,
                 aux_loss_fn=None, logger=None, csv_logger=None):
        """Legacy 2-task constructor (forecasting + anomaly). Preserved for
        byte-compatibility with the committed FC+AD runs. For arbitrary task
        subsets use `MultiTaskFederatedTrainer.from_tasks(...)`."""
        fc = _TaskSpec("forecasting", "forecast", forecasting_model,
                       kind="forecast")
        an = _TaskSpec("anomaly", "anomaly", anomaly_model,
                       kind="reconstruct")
        self._setup([fc, an], device, config, aux_loss_fn, logger, csv_logger)

    @classmethod
    def from_tasks(cls, tasks, device, config, aux_loss_fn=None, logger=None,
                   csv_logger=None):
        """Generic constructor over an ordered `_TaskSpec` list."""
        obj = cls.__new__(cls)
        obj._setup(tasks, device, config, aux_loss_fn, logger, csv_logger)
        return obj

    def _setup(self, tasks, device, config, aux_loss_fn, logger, csv_logger):
        self.tasks = tasks
        self._task_by_name = {t.name: t for t in tasks}
        self.device = device
        self.config = config
        self.aux_loss_fn = aux_loss_fn
        self.log = logger or logging.getLogger(__name__)
        self.csv_logger = csv_logger

        self.aggregation_mode = getattr(config, 'aggregation_mode', 'dual')
        self.fl_strategy = getattr(config, 'fl_strategy', 'fedavg')
        self.fedprox_mu = getattr(config, 'fedprox_mu', 0.01)
        # SCAFFOLD: two control-update variants share all machinery below.
        #   "scaffold"    -> Option II: c_i+ = c_i - c + (x_global - x_trained)/(K*eta)
        #   "scaffold_c1" -> Option I:  c_i+ = mean grad f_i(x_global)
        self.use_scaffold = self.fl_strategy in ("scaffold", "scaffold_c1")
        self.scaffold_option = 1 if self.fl_strategy == "scaffold_c1" else 2
        # Selective backbone sharing (dual only): which backbone modules are
        # cross-task-shared. None => whole backbone == original dual.
        self.cross_shared_modules = (getattr(config, 'cross_shared_modules', None)
                                     or BACKBONE_PREFIXES)

        # Move each task's global model to the primary device.
        for t in self.tasks:
            t.model = t.model.to(device)

        # Limit federated training to a configurable number of GPUs.
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        max_federated_gpus = max(1, getattr(self.config, "max_federated_gpus", 1))
        usable_gpus = min(n_gpus, max_federated_gpus)
        if usable_gpus >= 2:
            self.devices = [torch.device(f"cuda:{i}") for i in range(usable_gpus)]
            self.log.info("Using %d GPUs for parallel client training: %s",
                          usable_gpus, self.devices)
        else:
            self.devices = [device]
            if n_gpus >= 1 and device.type == "cuda":
                self.log.info(
                    "Single-GPU mode on %s (available: %d, max: %d)",
                    device, n_gpus, max_federated_gpus)

        self._verify_backbone_compatibility()

        self.history = {"round": [], "round_seconds": []}
        for t in self.tasks:
            self.history[f"{t.csv_key}_test_loss"] = []

    # ── Backward-compatible model accessors ──

    @property
    def forecasting_model(self):
        return self._task_by_name["forecasting"].model

    @property
    def anomaly_model(self):
        return self._task_by_name["anomaly"].model

    def _model_by_name(self, name):
        return self._task_by_name[name].model

    # SCAFFOLD control accessors under the legacy per-task names (used by
    # validate_scaffold.py). Valid for the compat 2-task registry.
    @property
    def global_control_fc(self):
        return self._task_by_name["forecasting"].global_control

    @property
    def client_controls_fc(self):
        return self._task_by_name["forecasting"].client_controls

    @property
    def global_control_an(self):
        return self._task_by_name["anomaly"].global_control

    @property
    def client_controls_an(self):
        return self._task_by_name["anomaly"].client_controls

    def _verify_backbone_compatibility(self):
        """Ensure all tasks' models have identical backbone parameter shapes."""
        ref = self.tasks[0].model.state_dict()
        for t in self.tasks[1:]:
            st = t.model.state_dict()
            for name in ref:
                if _is_backbone(name):
                    assert name in st, (
                        f"Backbone param '{name}' missing from task '{t.name}'")
                    assert ref[name].shape == st[name].shape, (
                        f"Shape mismatch for '{name}' in task '{t.name}': "
                        f"{ref[name].shape} vs {st[name].shape}")
        self.log.info("Backbone compatibility verified across %d tasks.",
                      len(self.tasks))

    def _compute_aux_loss(self, model):
        if self.aux_loss_fn is not None:
            return self.aux_loss_fn(model)
        return 0.0

    def _save_latest_checkpoint(self, round_num):
        """Overwrite per-round 'latest' snapshots of each task's global model so a
        mid-condition crash preserves the last completed round. Failure here never
        aborts training."""
        try:
            ckpt_dir = self.config.checkpoint_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            tag = getattr(self.config, "cohort_tag", "")
            prefix = f"fed_{self.aggregation_mode}_{self.fl_strategy}{tag}"
            for t in self.tasks:
                torch.save(t.model.state_dict(),
                           os.path.join(ckpt_dir, f"{prefix}_{t.name}_latest.pt"))
        except Exception as e:  # crash insurance must never crash the run itself
            self.log.warning("  per-round 'latest' checkpoint save failed "
                             "(round %d): %s", round_num + 1, e)

    # ── Local training (dispatched by task kind) ──

    def _train_forecast_kind(self, client_model, train_loader, target_device,
                             global_params=None, global_control=None,
                             client_control=None):
        """Train one forecasting-kind client. FedAvg/FedProx/SCAFFOLD."""
        client_model.train()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            client_model.parameters(),
            lr=self.config.client_lr,
            weight_decay=self.config.weight_decay,
        )
        total_loss, n_batches = 0.0, 0

        for epoch in range(self.config.local_epochs):
            for x, y in train_loader:
                x = x.float().to(target_device, non_blocking=True)
                y = y.float().to(target_device, non_blocking=True)
                optimizer.zero_grad()
                y_pred = client_model(x)
                loss = loss_fn(y_pred, y) + self._compute_aux_loss(client_model)

                if self.fl_strategy == "fedprox" and global_params is not None:
                    prox = sum(
                        ((p - global_params[n]) ** 2).sum()
                        for n, p in client_model.named_parameters()
                        if p.requires_grad)
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                loss.backward()

                if self.use_scaffold and global_control is not None:
                    for n, p in client_model.named_parameters():
                        if p.grad is not None and n in global_control:
                            p.grad.data.add_(
                                global_control[n] - client_control[n])

                nn.utils.clip_grad_norm_(client_model.parameters(),
                                         self.config.grad_clip)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return client_model, total_loss / max(n_batches, 1)

    def _train_reconstruct_kind(self, client_model, train_loader, target_device,
                                global_params=None, global_control=None,
                                client_control=None):
        """Train one reconstruct-kind client (anomaly / imputation): random mask,
        loss on masked positions only. FedAvg/FedProx/SCAFFOLD."""
        client_model.train()
        loss_fn = nn.MSELoss(reduction='none')
        optimizer = torch.optim.AdamW(
            client_model.parameters(),
            lr=self.config.client_lr,
            weight_decay=self.config.weight_decay,
        )
        mask_rate = self.config.mask_rate
        total_loss, n_batches = 0.0, 0

        for epoch in range(self.config.local_epochs):
            for sequences, labels in train_loader:
                sequences = sequences.float().to(target_device,
                                                  non_blocking=True)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)

                mask = (torch.rand_like(sequences) > mask_rate).float()
                x_masked = sequences * mask
                optimizer.zero_grad()

                recon = client_model(x_masked, x_mask=mask)

                per_elem = loss_fn(recon, sequences)
                masked_pos = (mask == 0)
                loss = (per_elem[masked_pos].mean() if masked_pos.any()
                        else per_elem.mean())
                loss = loss + self._compute_aux_loss(client_model)

                if self.fl_strategy == "fedprox" and global_params is not None:
                    prox = sum(
                        ((p - global_params[n]) ** 2).sum()
                        for n, p in client_model.named_parameters()
                        if p.requires_grad)
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                loss.backward()

                if self.use_scaffold and global_control is not None:
                    for n, p in client_model.named_parameters():
                        if p.grad is not None and n in global_control:
                            p.grad.data.add_(
                                global_control[n] - client_control[n])

                nn.utils.clip_grad_norm_(client_model.parameters(),
                                         self.config.grad_clip)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return client_model, total_loss / max(n_batches, 1)

    def _train_fn_for(self, kind):
        return (self._train_forecast_kind if kind == "forecast"
                else self._train_reconstruct_kind)

    # ── SCAFFOLD Option I: control = mean gradient at the global model ──

    def _grad_forecast_kind(self, model, train_loader, target_device):
        """SCAFFOLD Option I control (forecast kind): mean gradient of the loss
        over the client's data at the model's CURRENT (== global) params. No
        optimizer step. Returns {name: grad} on CPU; params unchanged."""
        model.train()
        loss_fn = nn.MSELoss()
        model.zero_grad(set_to_none=True)
        n_batches = 0
        for x, y in train_loader:
            x = x.float().to(target_device, non_blocking=True)
            y = y.float().to(target_device, non_blocking=True)
            loss = loss_fn(model(x), y) + self._compute_aux_loss(model)
            loss.backward()
            n_batches += 1
        denom = max(n_batches, 1)
        grads = {n: (p.grad.detach().div(denom).cpu() if p.grad is not None
                     else torch.zeros_like(p, device="cpu"))
                 for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        return grads

    def _grad_reconstruct_kind(self, model, train_loader, target_device):
        """SCAFFOLD Option I control (reconstruct kind): mean gradient of the
        masked-reconstruct loss over the client's data at current params."""
        model.train()
        loss_fn = nn.MSELoss(reduction='none')
        mask_rate = self.config.mask_rate
        model.zero_grad(set_to_none=True)
        n_batches = 0
        for sequences, labels in train_loader:
            sequences = sequences.float().to(target_device, non_blocking=True)
            if sequences.dim() == 2:
                sequences = sequences.unsqueeze(-1)
            mask = (torch.rand_like(sequences) > mask_rate).float()
            recon = model(sequences * mask, x_mask=mask)
            per_elem = loss_fn(recon, sequences)
            masked_pos = (mask == 0)
            loss = (per_elem[masked_pos].mean() if masked_pos.any()
                    else per_elem.mean())
            loss = loss + self._compute_aux_loss(model)
            loss.backward()
            n_batches += 1
        denom = max(n_batches, 1)
        grads = {n: (p.grad.detach().div(denom).cpu() if p.grad is not None
                     else torch.zeros_like(p, device="cpu"))
                 for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        return grads

    def _grad_fn_for(self, kind):
        return (self._grad_forecast_kind if kind == "forecast"
                else self._grad_reconstruct_kind)

    # Backward-compat aliases: the pre-refactor per-task method names, kept so
    # external scripts (analyze_update_conflict.py, older utilities) that grab
    # e.g. `trainer._train_forecasting_client_on` keep working. Same signatures.
    _train_forecasting_client_on = _train_forecast_kind
    _train_anomaly_client_on = _train_reconstruct_kind
    _grad_at_params_forecasting = _grad_forecast_kind
    _grad_at_params_anomaly = _grad_reconstruct_kind

    # ── Aggregation methods ──

    def _is_cross_shared(self, name):
        """True if a (backbone) param is cross-task shared under the active
        sharing scheme. Non-backbone params are never cross-shared (heads)."""
        return _is_backbone(name) and any(
            name.startswith(p) for p in self.cross_shared_modules)

    @staticmethod
    def _avg(name, states, weights, ref):
        """Weighted average of `name` over `states`; copy for int buffers."""
        if not ref[name].is_floating_point():
            return states[0][name]
        agg = torch.zeros_like(ref[name])
        for sd, w in zip(states, weights):
            agg += w * sd[name]
        return agg

    def _dual_aggregate(self):
        """Dual FedAvg (proposed), generic over the task registry, with optional
        SELECTIVE sharing:
          1. Cross-shared backbone modules: weighted avg across ALL clients of
             ALL tasks.
          2. Everything else, per task: weighted avg WITHIN that task's client
             group (task-private backbone modules + that task's heads).
        For the 2-task [forecasting, anomaly] registry with whole-backbone
        sharing this is byte-identical to the original `dual`.
        """
        # Per-task client states + weights, in task order.
        per_task_states = [[m.state_dict() for m in t._round_models]
                           for t in self.tasks]
        per_task_weights = [list(t._round_weights) for t in self.tasks]

        # All clients (task order preserved) for the cross-shared backbone.
        all_states, all_weights_raw = [], []
        for states, weights in zip(per_task_states, per_task_weights):
            all_states.extend(states)
            all_weights_raw.extend(weights)
        total = sum(all_weights_raw)
        all_weights = [w / total for w in all_weights_raw]

        ref = self.tasks[0].model.state_dict()  # backbone shapes shared
        global_shared = {name: self._avg(name, all_states, all_weights, ref)
                         for name in ref if self._is_cross_shared(name)}

        # Per-task private params (everything not cross-shared).
        for t, states, weights in zip(self.tasks, per_task_states,
                                      per_task_weights):
            tot = sum(weights)
            wn = [w / tot for w in weights]
            tstate = t.model.state_dict()
            private = {name: self._avg(name, states, wn, tstate)
                       for name in tstate if name not in global_shared}
            t.model.load_state_dict({**global_shared, **private})

    def _single_task_aggregate(self):
        """FedAvg ALL params within each task group. No cross-task sharing."""
        for t in self.tasks:
            states = [m.state_dict() for m in t._round_models]
            weights = list(t._round_weights)
            tot = sum(weights)
            wn = [w / tot for w in weights]
            tstate = t.model.state_dict()
            new_state = {name: self._avg(name, states, wn, tstate)
                         for name in tstate}
            t.model.load_state_dict(new_state)

    def _local_only_aggregate(self):
        """Average within-task models for EVALUATION only (not fed back)."""
        self._single_task_aggregate()

    # ── SCAFFOLD control variate management ──

    def _init_scaffold(self):
        """Initialize SCAFFOLD control variates (all zeros) per task."""
        for t in self.tasks:
            t.global_control = {
                n: torch.zeros_like(p).cpu()
                for n, p in t.model.named_parameters()}
            t.client_controls = {}
        self.log.info("SCAFFOLD control variates initialized (%d tasks).",
                      len(self.tasks))

    def _update_scaffold_global_controls(self, old_controls):
        """Update each task's global control: c += (1/S) * sum(c_i_new - c_i_old).

        `old_controls[task.name]` is the round-start snapshot of that task's
        per-client controls. A client seen for the FIRST time this round had
        control 0 at round start (lazily created during training) — treat a
        missing snapshot as zero so its control mass isn't dropped (keeps the
        SCAFFOLD invariant c == mean(c_i))."""
        for t in self.tasks:
            old = old_controls.get(t.name, {})
            for name in t.global_control:
                delta_sum = torch.zeros_like(t.global_control[name])
                n = 0
                for cid, ctrl in t.client_controls.items():
                    new_c = ctrl[name]
                    old_c = old.get(cid, {}).get(name)
                    if old_c is None:
                        old_c = torch.zeros_like(new_c)
                    delta_sum += (new_c - old_c)
                    n += 1
                if n > 0:
                    t.global_control[name] += delta_sum / n

    # ── Evaluation helpers ──

    def _eval_loss(self, task):
        """Per-round monitoring loss for a task (test MSE)."""
        model = task.model
        model.eval()
        losses = []
        loss_fn = nn.MSELoss()
        loader = task.test_loader
        with torch.no_grad():
            if task.kind == "forecast":
                for x, y in loader:
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    losses.append(loss_fn(model(x), y).item())
            else:
                for sequences, labels in loader:
                    sequences = sequences.float().to(self.device)
                    if sequences.dim() == 2:
                        sequences = sequences.unsqueeze(-1)
                    losses.append(loss_fn(model(sequences), sequences).item())
        return np.mean(losses) if losses else float("inf")

    def evaluate_forecasting(self, test_loader, model=None):
        """Return predictions and targets from a forecasting model."""
        model = model or self._model_by_name("forecasting")
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                all_preds.append(model(x).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_targets)

    def evaluate_anomaly_detection(self, data_loader, model=None):
        """Return anomaly scores and labels from an anomaly model."""
        model = model or self._model_by_name("anomaly")
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)
                recon = model(sequences)
                point_errors = (recon - sequences) ** 2
                window_scores = point_errors.mean(
                    dim=tuple(range(1, point_errors.dim())))
                all_scores.extend(window_scores.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist()
                                  if hasattr(labels, 'numpy')
                                  else [labels])
        return np.array(all_scores), np.array(all_labels)

    def _make_eval_mask(self, shape, mask_rate, pattern, rng):
        """Deterministic eval mask (1=keep, 0=masked). shape=(B,L,C).
        `pattern`: 'random' point masking or 'block' contiguous-span masking
        (one random span per window along the time axis)."""
        B, L, C = shape
        if pattern == "block":
            span = max(1, int(round(mask_rate * L)))
            mask = np.ones((B, L, C), dtype=np.float32)
            for b in range(B):
                start = rng.randint(0, max(1, L - span + 1))
                mask[b, start:start + span, :] = 0.0
            return torch.from_numpy(mask)
        # random point masking
        m = (rng.rand(B, L, C) > mask_rate).astype(np.float32)
        return torch.from_numpy(m)

    def evaluate_imputation(self, test_loader, model=None, mask_rate=None,
                            mask_pattern="random", seed=1234):
        """Impute masked values on the test windows and return the (true, pred)
        values AT MASKED POSITIONS ONLY, as flat numpy arrays. The mask is drawn
        deterministically (seeded, in dataloader order) so the metric is
        reproducible. Feed the result to compute_imputation_metrics."""
        model = model or self._model_by_name("imputation")
        model.eval()
        mr = self.config.mask_rate if mask_rate is None else mask_rate
        rng = np.random.RandomState(seed)
        trues, preds = [], []
        with torch.no_grad():
            for sequences, _labels in test_loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)
                mask = self._make_eval_mask(
                    tuple(sequences.shape), mr, mask_pattern, rng
                ).to(self.device)
                recon = model(sequences * mask, x_mask=mask)
                masked_pos = (mask == 0)
                if masked_pos.any():
                    trues.append(sequences[masked_pos].cpu().numpy())
                    preds.append(recon[masked_pos].cpu().numpy())
        if not trues:
            return np.array([]), np.array([])
        return np.concatenate(trues), np.concatenate(preds)

    # ── Client training helper ──

    def _train_all_clients_on_device(self, task, target_device):
        """Train all clients of `task` on a single device. Handles all FL
        strategies (fedavg/fedprox/scaffold/scaffold_c1) and local_only."""
        group_start = time.time()
        global_model = task.model
        client_loaders = task.client_loaders
        client_ids = list(client_loaders.keys())
        train_fn = self._train_fn_for(task.kind)
        grad_fn = self._grad_fn_for(task.kind)
        persistent_clients = task.persistent
        models, weights = [], []

        # FedProx: snapshot global params once per group
        global_params = None
        if self.fl_strategy == "fedprox":
            global_params = {n: p.data.clone().to(target_device)
                            for n, p in global_model.named_parameters()}

        # SCAFFOLD: get global control and per-client controls dict
        global_control = None
        controls_dict = None
        if self.use_scaffold:
            global_control = {n: v.to(target_device)
                              for n, v in task.global_control.items()}
            controls_dict = task.client_controls

        for cid in client_ids:
            cdata = client_loaders[cid]
            if cdata["train"] is None:
                continue

            # Initialize client model
            if persistent_clients is not None and cid in persistent_clients:
                client = copy.deepcopy(global_model).to(target_device)
                saved = persistent_clients[cid]
                client.load_state_dict(
                    {k: v.to(target_device) for k, v in saved.items()})
            else:
                client = copy.deepcopy(global_model).to(target_device)

            # SCAFFOLD: get per-client control
            client_control = None
            if self.use_scaffold:
                if cid not in controls_dict:
                    controls_dict[cid] = {
                        n: torch.zeros_like(p).cpu()
                        for n, p in global_model.named_parameters()}
                client_control = {n: v.to(target_device)
                                  for n, v in controls_dict[cid].items()}

            # SCAFFOLD Option I: recompute the control as the mean gradient at the
            # global model, on `client` BEFORE it is trained (currently == the
            # global model). Isolate this extra pass from training's RNG stream
            # (dropout, the mask) by save/restoring generator state, so Option I
            # diverges from FedAvg ONLY through the control variates.
            option1_control = None
            if self.use_scaffold and self.scaffold_option == 1:
                cpu_rng = torch.get_rng_state()
                cuda_rng = None
                dev_t = torch.device(target_device)
                if dev_t.type == "cuda":
                    cuda_rng = torch.cuda.get_rng_state(dev_t)
                option1_control = grad_fn(client, cdata["train"], target_device)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, dev_t)

            # Train client
            trained, loss = train_fn(
                client, cdata["train"], target_device,
                global_params=global_params,
                global_control=global_control,
                client_control=client_control)

            # SCAFFOLD: update client control variate
            if self.use_scaffold and self.scaffold_option == 1:
                controls_dict[cid] = option1_control
            elif self.use_scaffold:
                n_steps = self.config.local_epochs * len(cdata["train"])
                lr = self.config.client_lr
                new_control = {}
                trained_state = trained.state_dict()
                for name, p_global in global_model.named_parameters():
                    p_trained = trained_state[name].to(target_device)
                    c_i = client_control[name]
                    c = global_control[name]
                    new_control[name] = (
                        c_i - c
                        + (p_global.data.to(target_device) - p_trained)
                        / (n_steps * lr)
                    ).cpu()
                controls_dict[cid] = new_control

            # Move to primary device for aggregation
            trained = trained.to(self.device)
            models.append(trained)
            weights.append(cdata["n_samples"])

            # Local-only: persist client state_dict for next round (CPU tensors)
            if persistent_clients is not None:
                persistent_clients[cid] = {
                    k: v.cpu().clone() for k, v in trained.state_dict().items()
                }

        self.log.info("  [%s] %d clients trained on %s in %.0fs",
                      task.name, len(models), target_device,
                      time.time() - group_start)
        return models, weights

    # ── Main training loop ──

    def train(self, forecast_client_loaders, anomaly_client_loaders,
              forecast_test_loader, anomaly_test_loader):
        """Legacy 2-task entry point (forecasting + anomaly). Assigns loaders to
        the compat tasks and runs the generic loop; returns history with the
        legacy keys (`forecast_test_loss`, `anomaly_test_loss`)."""
        self._task_by_name["forecasting"].client_loaders = forecast_client_loaders
        self._task_by_name["forecasting"].test_loader = forecast_test_loader
        self._task_by_name["anomaly"].client_loaders = anomaly_client_loaders
        self._task_by_name["anomaly"].test_loader = anomaly_test_loader
        return self._run()

    def train_tasks(self):
        """Generic entry point: loaders already attached to each _TaskSpec (via
        from_tasks + assignment)."""
        return self._run()

    def _run(self):
        """Run multi-task federated training over the attached task registry."""
        total_clients = sum(len(t.client_loaders) for t in self.tasks)

        self.log.info("Multi-Task Federated Training")
        self.log.info("  Mode: %s | Strategy: %s",
                      self.aggregation_mode, self.fl_strategy)
        for t in self.tasks:
            self.log.info("  Task %-12s (%s): %d clients",
                          t.name, t.kind, len(t.client_loaders))
        self.log.info("  Total clients:       %d", total_clients)
        self.log.info("  Rounds: %d, Local epochs: %d",
                      self.config.num_rounds, self.config.local_epochs)
        self.log.info("  GPUs: %d", len(self.devices))
        if self.fl_strategy == "fedprox":
            self.log.info("  FedProx mu: %.4f", self.fedprox_mu)

        if self.use_scaffold:
            self._init_scaffold()

        # local_only: persistent per-client state per task
        if self.aggregation_mode == "local_only":
            for t in self.tasks:
                t.persistent = {}

        use_parallel = len(self.devices) >= 2

        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            self.log.info("Round %d/%d | %d clients",
                         round_num + 1, self.config.num_rounds, total_clients)

            # Snapshot SCAFFOLD controls for delta computation (per task)
            old_controls = {}
            if self.use_scaffold:
                for t in self.tasks:
                    old_controls[t.name] = {
                        cid: {n: v.clone() for n, v in ctrl.items()}
                        for cid, ctrl in t.client_controls.items()}

            # ── Train all clients ──
            if use_parallel:
                with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
                    futures = {}
                    for i, t in enumerate(self.tasks):
                        dev = self.devices[i % len(self.devices)]
                        futures[t.name] = ex.submit(
                            self._train_all_clients_on_device, t, dev)
                    for t in self.tasks:
                        t._round_models, t._round_weights = \
                            futures[t.name].result()
            else:
                for t in self.tasks:
                    t._round_models, t._round_weights = \
                        self._train_all_clients_on_device(t, self.device)

            # ── Aggregation ──
            if self.aggregation_mode == "dual":
                self._dual_aggregate()
            elif self.aggregation_mode == "single_task":
                self._single_task_aggregate()
            elif self.aggregation_mode == "local_only":
                self._local_only_aggregate()

            if self.use_scaffold:
                self._update_scaffold_global_controls(old_controls)

            # ── Evaluate (every eval_every rounds; always on the final round) ──
            eval_every = max(1, getattr(self.config, "eval_every", 1))
            is_last = (round_num + 1) == self.config.num_rounds
            if (round_num + 1) % eval_every == 0 or is_last:
                losses = {t.name: self._eval_loss(t) for t in self.tasks}
                round_seconds = time.time() - round_start

                self.history["round"].append(round_num + 1)
                self.history["round_seconds"].append(round_seconds)
                for t in self.tasks:
                    self.history[f"{t.csv_key}_test_loss"].append(
                        losses[t.name])

                msg = " | ".join(f"{t.name} test MSE: {losses[t.name]:.6f}"
                                 for t in self.tasks)
                self.log.info("  %s | round %.0fs", msg, round_seconds)

                if self.csv_logger is not None:
                    row = {"round": round_num + 1,
                           "round_seconds": f"{round_seconds:.1f}"}
                    for t in self.tasks:
                        row[f"{t.csv_key}_test_mse"] = f"{losses[t.name]:.6f}"
                    self.csv_logger.log(row)

            # Per-round crash insurance
            self._save_latest_checkpoint(round_num)

            # Free client models
            for t in self.tasks:
                t._round_models = None
            for dev in self.devices:
                if dev.type == "cuda":
                    with torch.cuda.device(dev):
                        torch.cuda.empty_cache()

        return self.history
