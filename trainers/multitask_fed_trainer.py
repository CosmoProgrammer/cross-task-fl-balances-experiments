"""
Multi-Task Federated Trainer with configurable aggregation and FL strategies.

Aggregation modes:
  - "dual":        Backbone across ALL 70 clients, heads within task (proposed)
  - "single_task": FedAvg ALL params within each task group independently
  - "local_only":  No aggregation fed back; clients train from own checkpoints

FL strategies:
  - "fedavg":   Standard Federated Averaging
  - "fedprox":  FedAvg + proximal term (Li et al., 2020)

Architecture split (MambaMixer / SSD-Net):
  Backbone: patch_encoders, patch_decoders, cross_scale_gates
  Heads:    pred_heads, scale_router

After training:
  - Global Forecasting Model = aggregated backbone + aggregated forecasting heads
  - Global Anomaly Model    = aggregated backbone + aggregated anomaly heads
"""
import copy
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from models.mamba_mixer import MambaMixer


# Parameter classification
BACKBONE_PREFIXES = ("patch_encoders.", "patch_decoders.", "cross_scale_gates.")


def _is_backbone(name):
    return any(name.startswith(p) for p in BACKBONE_PREFIXES)


class MultiTaskFederatedTrainer:
    """Federated trainer with configurable aggregation and strategies."""

    def __init__(self, forecasting_model, anomaly_model, device, config,
                 aux_loss_fn=None, logger=None, csv_logger=None):
        """
        Args:
            forecasting_model: MambaMixer(out_len=24)  -- global forecasting model
            anomaly_model:     MambaMixer(out_len=128) -- global anomaly model
            device: torch device (primary)
            config: ExperimentConfig
            aux_loss_fn: optional aux loss (called as aux_loss_fn(model))
            logger: Python logger instance
            csv_logger: CSVMetricsLogger for per-round tracking
        """
        self.device = device
        self.config = config
        self.aux_loss_fn = aux_loss_fn
        self.log = logger or logging.getLogger(__name__)
        self.csv_logger = csv_logger

        self.aggregation_mode = getattr(config, 'aggregation_mode', 'dual')
        self.fl_strategy = getattr(config, 'fl_strategy', 'fedavg')
        self.fedprox_mu = getattr(config, 'fedprox_mu', 0.01)

        self.forecasting_model = forecasting_model.to(device)
        self.anomaly_model = anomaly_model.to(device)

        # Limit federated training to a configurable number of GPUs.
        # Defaulting to a single GPU avoids fragile threaded CUDA execution.
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

        # Verify backbone compatibility
        self._verify_backbone_compatibility()

        self.history = {
            "round": [],
            "forecast_test_loss": [],
            "anomaly_test_loss": [],
        }

    def _verify_backbone_compatibility(self):
        """Ensure both models have identical backbone parameter shapes."""
        fc_state = self.forecasting_model.state_dict()
        an_state = self.anomaly_model.state_dict()
        for name in fc_state:
            if _is_backbone(name):
                assert name in an_state, (
                    f"Backbone param '{name}' missing from anomaly model")
                assert fc_state[name].shape == an_state[name].shape, (
                    f"Shape mismatch for '{name}': "
                    f"{fc_state[name].shape} vs {an_state[name].shape}")
        self.log.info("Backbone compatibility verified.")

    def _compute_aux_loss(self, model):
        if self.aux_loss_fn is not None:
            return self.aux_loss_fn(model)
        return 0.0

    # ── Local training ──

    def _train_forecasting_client_on(self, client_model, train_loader,
                                     target_device, global_params=None,
                                     global_control=None, client_control=None):
        """Train one forecasting client. Supports FedAvg/FedProx/SCAFFOLD."""
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

                # FedProx: proximal term  mu/2 * ||w - w_global||^2
                if self.fl_strategy == "fedprox" and global_params is not None:
                    prox = sum(
                        ((p - global_params[n]) ** 2).sum()
                        for n, p in client_model.named_parameters()
                        if p.requires_grad)
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                loss.backward()

                # SCAFFOLD: gradient correction  g_corrected = g - c_i + c
                if self.fl_strategy == "scaffold" and global_control is not None:
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

    def _train_anomaly_client_on(self, client_model, train_loader,
                                 target_device, global_params=None,
                                 global_control=None, client_control=None):
        """Train one anomaly client. Supports FedAvg/FedProx/SCAFFOLD."""
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

                # FedProx: proximal term
                if self.fl_strategy == "fedprox" and global_params is not None:
                    prox = sum(
                        ((p - global_params[n]) ** 2).sum()
                        for n, p in client_model.named_parameters()
                        if p.requires_grad)
                    loss = loss + (self.fedprox_mu / 2.0) * prox

                loss.backward()

                # SCAFFOLD: gradient correction
                if self.fl_strategy == "scaffold" and global_control is not None:
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

    # ── Aggregation methods ──

    def _dual_aggregate(self, forecast_clients, forecast_weights,
                        anomaly_clients, anomaly_weights):
        """
        Dual FedAvg aggregation (proposed method):
          1. Backbone: weighted average across ALL 70 clients
          2. Forecasting heads: weighted average across 35 forecasting clients
          3. Anomaly heads: weighted average across 35 anomaly clients
        """
        all_weights_raw = forecast_weights + anomaly_weights
        total = sum(all_weights_raw)
        all_weights = [w / total for w in all_weights_raw]

        fc_total = sum(forecast_weights)
        fc_weights_norm = [w / fc_total for w in forecast_weights]
        an_total = sum(anomaly_weights)
        an_weights_norm = [w / an_total for w in anomaly_weights]

        # Cache state_dicts (avoids repeated allocation in inner loops)
        fc_client_states = [m.state_dict() for m in forecast_clients]
        an_client_states = [m.state_dict() for m in anomaly_clients]
        all_client_states = fc_client_states + an_client_states

        # 1. Aggregate backbone across ALL 70 clients
        global_backbone = {}
        fc_state = self.forecasting_model.state_dict()
        for name in fc_state:
            if not _is_backbone(name):
                continue
            if fc_state[name].is_floating_point():
                agg = torch.zeros_like(fc_state[name])
                for sd, w in zip(all_client_states, all_weights):
                    agg += w * sd[name]
                global_backbone[name] = agg
            else:
                global_backbone[name] = all_client_states[0][name]

        # 2. Aggregate forecasting heads across 35 forecasting clients
        global_fc_heads = {}
        for name in fc_state:
            if _is_backbone(name):
                continue
            if fc_state[name].is_floating_point():
                agg = torch.zeros_like(fc_state[name])
                for sd, w in zip(fc_client_states, fc_weights_norm):
                    agg += w * sd[name]
                global_fc_heads[name] = agg
            else:
                global_fc_heads[name] = fc_client_states[0][name]

        # 3. Aggregate anomaly heads across 35 anomaly clients
        an_state = self.anomaly_model.state_dict()
        global_an_heads = {}
        for name in an_state:
            if _is_backbone(name):
                continue
            if an_state[name].is_floating_point():
                agg = torch.zeros_like(an_state[name])
                for sd, w in zip(an_client_states, an_weights_norm):
                    agg += w * sd[name]
                global_an_heads[name] = agg
            else:
                global_an_heads[name] = an_client_states[0][name]

        # 4. Load into global models
        self.forecasting_model.load_state_dict(
            {**global_backbone, **global_fc_heads})
        self.anomaly_model.load_state_dict(
            {**global_backbone, **global_an_heads})

    def _single_task_aggregate(self, forecast_clients, forecast_weights,
                               anomaly_clients, anomaly_weights):
        """FedAvg ALL params within each task group. No cross-task sharing."""
        # Cache state_dicts once
        fc_client_states = [m.state_dict() for m in forecast_clients]
        an_client_states = [m.state_dict() for m in anomaly_clients]

        # Forecasting: average all params across forecasting clients only
        fc_total = sum(forecast_weights)
        fc_weights_norm = [w / fc_total for w in forecast_weights]
        fc_state = self.forecasting_model.state_dict()
        new_fc = {}
        for name in fc_state:
            if fc_state[name].is_floating_point():
                agg = torch.zeros_like(fc_state[name])
                for sd, w in zip(fc_client_states, fc_weights_norm):
                    agg += w * sd[name]
                new_fc[name] = agg
            else:
                new_fc[name] = fc_client_states[0][name]
        self.forecasting_model.load_state_dict(new_fc)

        # Anomaly: average all params across anomaly clients only
        an_total = sum(anomaly_weights)
        an_weights_norm = [w / an_total for w in anomaly_weights]
        an_state = self.anomaly_model.state_dict()
        new_an = {}
        for name in an_state:
            if an_state[name].is_floating_point():
                agg = torch.zeros_like(an_state[name])
                for sd, w in zip(an_client_states, an_weights_norm):
                    agg += w * sd[name]
                new_an[name] = agg
            else:
                new_an[name] = an_client_states[0][name]
        self.anomaly_model.load_state_dict(new_an)

    def _local_only_aggregate(self, forecast_clients, forecast_weights,
                              anomaly_clients, anomaly_weights):
        """Average within-task models for EVALUATION only.

        The aggregated global models are used for metric computation but are
        NOT fed back into client initialization — each client continues from
        its own persistent checkpoint in the next round.
        """
        self._single_task_aggregate(forecast_clients, forecast_weights,
                                    anomaly_clients, anomaly_weights)

    # ── SCAFFOLD control variate management ──

    def _init_scaffold(self):
        """Initialize SCAFFOLD control variates (all zeros)."""
        self.global_control_fc = {
            n: torch.zeros_like(p).cpu()
            for n, p in self.forecasting_model.named_parameters()}
        self.global_control_an = {
            n: torch.zeros_like(p).cpu()
            for n, p in self.anomaly_model.named_parameters()}
        self.client_controls_fc = {}
        self.client_controls_an = {}
        self.log.info("SCAFFOLD control variates initialized.")

    def _update_scaffold_global_controls(self, fc_ids, an_ids,
                                         old_fc_controls, old_an_controls):
        """Update global control variates: c += (1/S) * sum(c_i_new - c_i_old)."""
        for name in self.global_control_fc:
            delta_sum = torch.zeros_like(self.global_control_fc[name])
            n = 0
            for cid in fc_ids:
                if cid in self.client_controls_fc and cid in old_fc_controls:
                    delta_sum += (self.client_controls_fc[cid][name]
                                  - old_fc_controls[cid][name])
                    n += 1
            if n > 0:
                self.global_control_fc[name] += delta_sum / n

        for name in self.global_control_an:
            delta_sum = torch.zeros_like(self.global_control_an[name])
            n = 0
            for cid in an_ids:
                if cid in self.client_controls_an and cid in old_an_controls:
                    delta_sum += (self.client_controls_an[cid][name]
                                  - old_an_controls[cid][name])
                    n += 1
            if n > 0:
                self.global_control_an[name] += delta_sum / n

    # ── Evaluation helpers ──

    def _eval_forecasting(self, loader):
        self.forecasting_model.eval()
        losses = []
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            for x, y in loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                losses.append(loss_fn(self.forecasting_model(x), y).item())
        return np.mean(losses) if losses else float("inf")

    def _eval_anomaly(self, loader):
        self.anomaly_model.eval()
        losses = []
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            for sequences, labels in loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)
                recon = self.anomaly_model(sequences)
                losses.append(loss_fn(recon, sequences).item())
        return np.mean(losses) if losses else float("inf")

    def evaluate_forecasting(self, test_loader):
        """Return predictions and targets from global forecasting model."""
        self.forecasting_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                all_preds.append(self.forecasting_model(x).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_targets)

    def evaluate_anomaly_detection(self, data_loader):
        """Return anomaly scores and labels from global anomaly model."""
        self.anomaly_model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)
                recon = self.anomaly_model(sequences)
                point_errors = (recon - sequences) ** 2
                window_scores = point_errors.mean(
                    dim=tuple(range(1, point_errors.dim())))
                all_scores.extend(window_scores.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist()
                                  if hasattr(labels, 'numpy')
                                  else [labels])
        return np.array(all_scores), np.array(all_labels)

    # ── Client training helper ──

    def _train_all_clients_on_device(self, client_ids, client_loaders,
                                     global_model, train_fn, target_device,
                                     task="forecasting",
                                     persistent_clients=None):
        """Train all clients in a group on a single device.

        Handles all FL strategies (fedavg, fedprox, scaffold) and supports
        local_only mode via persistent_clients dict.
        """
        models, weights = [], []

        # FedProx: snapshot global params once per group
        global_params = None
        if self.fl_strategy == "fedprox":
            global_params = {n: p.data.clone().to(target_device)
                            for n, p in global_model.named_parameters()}

        # SCAFFOLD: get global control and per-client controls dict
        global_control = None
        controls_dict = None
        if self.fl_strategy == "scaffold":
            gc = (self.global_control_fc if task == "forecasting"
                  else self.global_control_an)
            global_control = {n: v.to(target_device) for n, v in gc.items()}
            controls_dict = (self.client_controls_fc if task == "forecasting"
                            else self.client_controls_an)

        for cid in client_ids:
            cdata = client_loaders[cid]
            if cdata["train"] is None:
                continue

            # Initialize client model
            if persistent_clients is not None and cid in persistent_clients:
                # Local-only: create fresh model, load client's saved state
                client = copy.deepcopy(global_model).to(target_device)
                saved = persistent_clients[cid]
                client.load_state_dict(
                    {k: v.to(target_device) for k, v in saved.items()})
            else:
                client = copy.deepcopy(global_model).to(target_device)

            # SCAFFOLD: get per-client control
            client_control = None
            if self.fl_strategy == "scaffold":
                if cid not in controls_dict:
                    controls_dict[cid] = {
                        n: torch.zeros_like(p).cpu()
                        for n, p in global_model.named_parameters()}
                client_control = {n: v.to(target_device)
                                  for n, v in controls_dict[cid].items()}

            # Train client
            trained, loss = train_fn(
                client, cdata["train"], target_device,
                global_params=global_params,
                global_control=global_control,
                client_control=client_control)

            # SCAFFOLD: update client control variate (Option II)
            # c_i_new = c_i - c + (x_global - x_trained) / (K * eta)
            if self.fl_strategy == "scaffold":
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

            # Local-only: persist client state_dict for next round
            # (plain CPU tensors — avoids CUDA deepcopy which segfaults over many rounds)
            if persistent_clients is not None:
                persistent_clients[cid] = {
                    k: v.cpu().clone() for k, v in trained.state_dict().items()
                }

        return models, weights

    # ── Main training loop ──

    def train(self, forecast_client_loaders, anomaly_client_loaders,
              forecast_test_loader, anomaly_test_loader):
        """Run multi-task federated training.

        Args:
            forecast_client_loaders: {bid: {"train": loader, "n_samples": int}}
            anomaly_client_loaders:  {bid: {"train": loader, "n_samples": int}}
            forecast_test_loader: ASHRAE test buildings
            anomaly_test_loader:  LEAD test buildings

        Returns:
            history dict with per-round metrics
        """
        fc_ids = list(forecast_client_loaders.keys())
        an_ids = list(anomaly_client_loaders.keys())
        total_clients = len(fc_ids) + len(an_ids)

        self.log.info("Multi-Task Federated Training")
        self.log.info("  Mode: %s | Strategy: %s",
                      self.aggregation_mode, self.fl_strategy)
        self.log.info("  Forecasting clients: %d", len(fc_ids))
        self.log.info("  Anomaly clients:     %d", len(an_ids))
        self.log.info("  Total clients:       %d", total_clients)
        self.log.info("  Rounds: %d, Local epochs: %d",
                      self.config.num_rounds, self.config.local_epochs)
        self.log.info("  GPUs: %d", len(self.devices))
        if self.fl_strategy == "fedprox":
            self.log.info("  FedProx mu: %.4f", self.fedprox_mu)

        # Initialize SCAFFOLD control variates
        if self.fl_strategy == "scaffold":
            self._init_scaffold()

        # Initialize persistent client models for local_only mode
        persistent_fc = {} if self.aggregation_mode == "local_only" else None
        persistent_an = {} if self.aggregation_mode == "local_only" else None

        use_parallel = len(self.devices) >= 2

        for round_num in range(self.config.num_rounds):
            self.log.info("Round %d/%d | %d clients",
                         round_num + 1, self.config.num_rounds, total_clients)

            # Snapshot SCAFFOLD controls for delta computation
            old_fc_controls, old_an_controls = {}, {}
            if self.fl_strategy == "scaffold":
                old_fc_controls = {
                    cid: {n: v.clone() for n, v in
                          self.client_controls_fc[cid].items()}
                    for cid in fc_ids
                    if cid in self.client_controls_fc}
                old_an_controls = {
                    cid: {n: v.clone() for n, v in
                          self.client_controls_an[cid].items()}
                    for cid in an_ids
                    if cid in self.client_controls_an}

            # ── Train all clients ──
            if use_parallel:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fc_future = executor.submit(
                        self._train_all_clients_on_device,
                        fc_ids, forecast_client_loaders,
                        self.forecasting_model,
                        self._train_forecasting_client_on,
                        self.devices[0],
                        task="forecasting",
                        persistent_clients=persistent_fc)
                    an_future = executor.submit(
                        self._train_all_clients_on_device,
                        an_ids, anomaly_client_loaders,
                        self.anomaly_model,
                        self._train_anomaly_client_on,
                        self.devices[1],
                        task="anomaly",
                        persistent_clients=persistent_an)

                    forecast_models, forecast_weights = fc_future.result()
                    anomaly_models, anomaly_weights = an_future.result()
            else:
                forecast_models, forecast_weights = \
                    self._train_all_clients_on_device(
                        fc_ids, forecast_client_loaders,
                        self.forecasting_model,
                        self._train_forecasting_client_on,
                        self.device,
                        task="forecasting",
                        persistent_clients=persistent_fc)
                anomaly_models, anomaly_weights = \
                    self._train_all_clients_on_device(
                        an_ids, anomaly_client_loaders,
                        self.anomaly_model,
                        self._train_anomaly_client_on,
                        self.device,
                        task="anomaly",
                        persistent_clients=persistent_an)

            # ── Aggregation ──
            if self.aggregation_mode == "dual":
                self._dual_aggregate(forecast_models, forecast_weights,
                                     anomaly_models, anomaly_weights)
            elif self.aggregation_mode == "single_task":
                self._single_task_aggregate(forecast_models, forecast_weights,
                                            anomaly_models, anomaly_weights)
            elif self.aggregation_mode == "local_only":
                self._local_only_aggregate(forecast_models, forecast_weights,
                                           anomaly_models, anomaly_weights)

            # Update SCAFFOLD global controls
            if self.fl_strategy == "scaffold":
                self._update_scaffold_global_controls(
                    fc_ids, an_ids, old_fc_controls, old_an_controls)

            # ── Evaluate ──
            fc_loss = self._eval_forecasting(forecast_test_loader)
            an_loss = self._eval_anomaly(anomaly_test_loader)

            self.history["round"].append(round_num + 1)
            self.history["forecast_test_loss"].append(fc_loss)
            self.history["anomaly_test_loss"].append(an_loss)

            self.log.info("  Forecast test MSE: %.6f | Anomaly test MSE: %.6f",
                         fc_loss, an_loss)

            # CSV logging
            if self.csv_logger is not None:
                self.csv_logger.log({
                    "round": round_num + 1,
                    "forecast_test_mse": f"{fc_loss:.6f}",
                    "anomaly_test_mse": f"{an_loss:.6f}",
                })

            # Free client models
            del forecast_models, anomaly_models
            for dev in self.devices:
                if dev.type == "cuda":
                    with torch.cuda.device(dev):
                        torch.cuda.empty_cache()

        return self.history
