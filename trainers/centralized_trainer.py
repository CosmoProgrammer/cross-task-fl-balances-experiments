"""
Centralized training baselines for both forecasting and anomaly detection.

Used as a comparison against the multi-task FL approach:
  - Centralized forecasting: single model on pooled ASHRAE train buildings
  - Centralized anomaly: single model on pooled LEAD train buildings

Same train/test buildings as FL for fair comparison.
"""
import copy
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class CentralizedTrainer:
    """Centralized training baseline."""

    def __init__(self, model, device, config, aux_loss_fn=None, logger=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.aux_loss_fn = aux_loss_fn
        self.log = logger or logging.getLogger(__name__)
        self.best_val_loss = float("inf")
        self.best_state = None
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

    def _build_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.centralized_lr,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self, optimizer):
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=self.config.centralized_warmup)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(self.config.centralized_max_epochs
                      - self.config.centralized_warmup, 1),
            eta_min=self.config.centralized_min_lr,
        )
        return SequentialLR(optimizer, [warmup, cosine],
                            milestones=[self.config.centralized_warmup])

    def _compute_aux_loss(self):
        if self.aux_loss_fn is not None:
            return self.aux_loss_fn(self.model)
        return 0.0

    # ─── Forecasting ───

    def train_forecasting(self, train_loader, val_loader, csv_logger=None):
        loss_fn = nn.MSELoss()
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        for epoch in range(self.config.centralized_max_epochs):
            self.model.train()
            train_losses = []
            for x, y in train_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y) + self._compute_aux_loss()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config.grad_clip)
                optimizer.step()
                train_losses.append(loss.item())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step()

            avg_train = np.mean(train_losses)
            self.history["train_loss"].append(avg_train)

            val_loss = self._validate_forecasting(val_loader, loss_fn)
            self.history["val_loss"].append(val_loss)

            if csv_logger is not None:
                csv_logger.log({"epoch": epoch + 1,
                                "train_loss": f"{avg_train:.6f}",
                                "val_loss": f"{val_loss:.6f}"})

            if epoch % 10 == 0 or epoch == self.config.centralized_max_epochs - 1:
                self.log.info("Epoch %d/%d | train=%.6f val=%.6f",
                              epoch + 1, self.config.centralized_max_epochs,
                              avg_train, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.centralized_early_stop:
                    self.log.info("Early stopping at epoch %d", epoch + 1)
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def _validate_forecasting(self, val_loader, loss_fn):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                losses.append(loss_fn(self.model(x), y).item())
        return np.mean(losses) if losses else float("inf")

    def evaluate_forecasting(self, test_loader):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                all_preds.append(self.model(x).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_targets)

    # ─── Anomaly Detection (Mask-and-Reconstruct) ───

    def train_anomaly_detection(self, train_loader, val_loader, mask_rate=0.25,
                                csv_logger=None):
        loss_fn = nn.MSELoss(reduction='none')
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        for epoch in range(self.config.centralized_max_epochs):
            self.model.train()
            train_losses = []
            for sequences, labels in train_loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)

                mask = (torch.rand_like(sequences) > mask_rate).float()
                x_masked = sequences * mask
                optimizer.zero_grad()

                if hasattr(self.model, 'last_residual'):
                    recon = self.model(x_masked, x_mask=mask)
                else:
                    recon = self.model(x_masked)

                per_elem = loss_fn(recon, sequences)
                masked_pos = (mask == 0)
                loss = (per_elem[masked_pos].mean() if masked_pos.any()
                        else per_elem.mean())
                loss = loss + self._compute_aux_loss()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config.grad_clip)
                optimizer.step()
                train_losses.append(loss.item())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scheduler.step()

            avg_train = np.mean(train_losses)
            self.history["train_loss"].append(avg_train)

            val_loss = self._validate_anomaly(val_loader, mask_rate)
            self.history["val_loss"].append(val_loss)

            if csv_logger is not None:
                csv_logger.log({"epoch": epoch + 1,
                                "train_loss": f"{avg_train:.6f}",
                                "val_loss": f"{val_loss:.6f}"})

            if epoch % 10 == 0 or epoch == self.config.centralized_max_epochs - 1:
                self.log.info("Epoch %d/%d | train=%.6f val=%.6f",
                              epoch + 1, self.config.centralized_max_epochs,
                              avg_train, val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.centralized_early_stop:
                    self.log.info("Early stopping at epoch %d", epoch + 1)
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def _validate_anomaly(self, val_loader, mask_rate):
        self.model.eval()
        losses = []
        loss_fn = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)
                mask = (torch.rand_like(sequences) > mask_rate).float()
                x_masked = sequences * mask
                if hasattr(self.model, 'last_residual'):
                    recon = self.model(x_masked, x_mask=mask)
                else:
                    recon = self.model(x_masked)
                per_elem = loss_fn(recon, sequences)
                masked_pos = (mask == 0)
                loss = (per_elem[masked_pos].mean() if masked_pos.any()
                        else per_elem.mean())
                losses.append(loss.item())
        return np.mean(losses) if losses else float("inf")

    def evaluate_anomaly_detection(self, data_loader):
        self.model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.float().to(self.device)
                if sequences.dim() == 2:
                    sequences = sequences.unsqueeze(-1)
                if hasattr(self.model, 'last_residual'):
                    recon = self.model(sequences)
                else:
                    recon = self.model(sequences)
                point_errors = (recon - sequences) ** 2
                window_scores = point_errors.mean(
                    dim=tuple(range(1, point_errors.dim())))
                all_scores.extend(window_scores.cpu().numpy().tolist())
                all_labels.extend(
                    labels.numpy().tolist() if hasattr(labels, 'numpy')
                    else [labels] if isinstance(labels, int) else labels)
        return np.array(all_scores), np.array(all_labels)
