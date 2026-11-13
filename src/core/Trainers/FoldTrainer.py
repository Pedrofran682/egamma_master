import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.Callbacks.SPCallbackPyTorch import SPCallbackPyTorch

log = logging.getLogger()


class FoldTrainer:
    """Manages the training, validation, and callback evaluation loop for a single fold.

    Attributes:
        model: PyTorch neural network to train.
        optimizer: Optimization algorithm instance.
        loss_fn: Loss function criterion.
        device: PyTorch device on which computation occurs.
        pred_target_limiar: Decision boundary threshold for positive predictions.
        val_preds: Accumulated validation prediction probability arrays.
        val_targets: Accumulated validation target arrays.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        pred_target_limiar: float = 0.5,
    ) -> None:
        """Initializes FoldTrainer with execution components and hyperparameters.

        Args:
            model: PyTorch model instance.
            optimizer: Configured optimizer instance.
            loss_fn: Loss function module.
            device: Active computation device (CPU or GPU).
            pred_target_limiar: Decision threshold for classification (default 0.5).
        """
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn: nn.Module = loss_fn
        self.device: torch.device = device
        self.pred_target_limiar: float = pred_target_limiar
        self.val_preds: List[np.ndarray] = []
        self.val_targets: List[np.ndarray] = []

    def fit(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs: int,
        callback: SPCallbackPyTorch,
    ) -> Dict[str, Any]:
        """Runs the training and validation loop across epochs with early stopping callbacks.

        Args:
            train_dl: DataLoader providing training batches.
            val_dl: DataLoader providing validation batches.
            epochs: Total number of epochs to train.
            callback: SP metric tracker and early-stopping callback.

        Returns:
            Dictionary containing epoch metrics history, best weights, and SP/FA/PD scores.
        """
        fold_history: Dict[str, Any] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "callbackMetrics": [],
            "reapet": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch, train_dl)
            val_loss, val_acc = self._validate_epoch(epoch, val_dl)

            stop_training, callback_metrics = callback.on_epoch_end(
                self.model,
                epoch,
                self.val_targets,
                self.val_preds,
            )

            fold_history["train_loss"].append(train_loss)
            fold_history["train_acc"].append(train_acc)
            fold_history["val_loss"].append(val_loss)
            fold_history["val_acc"].append(val_acc)

            if stop_training:
                log.info(f"Early stopping triggered at epoch {epoch}")
                break

        fold_history["callbackMetrics"] = callback_metrics
        return {
            "history": fold_history,
            "best_weights": callback.get_best_model_weights(),
            "best_sp_value": callback.get_best_sp_value(),
            "best_fa_value": callback.get_best_fa_at_knee(),
            "best_pd_value": callback.get_best_pd_at_knee(),
        }

    def _train_epoch(self, epoch: int, dataloader: DataLoader) -> Tuple[float, float]:
        """Executes a single forward/backward training pass over the training dataloader.

        Args:
            epoch: Current epoch index.
            dataloader: DataLoader with training mini-batches.

        Returns:
            Tuple of (mean_epoch_loss, mean_epoch_accuracy).
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss, corrects = self._compute_batch_loss(batch, is_validation=False)
            loss.backward()
            self.optimizer.step()

            batch_size = batch[0].size(0)
            running_loss += loss.item() * batch_size
            running_corrects += corrects
            total_samples += batch_size

        return running_loss / total_samples, running_corrects / total_samples

    def _validate_epoch(self, epoch: int, dataloader: DataLoader) -> Tuple[float, float]:
        """Executes an evaluation pass over the validation dataloader without gradients.

        Args:
            epoch: Current epoch index.
            dataloader: DataLoader with validation mini-batches.

        Returns:
            Tuple of (mean_val_loss, mean_val_accuracy).
        """
        self.model.eval()
        self.val_preds = []
        self.val_targets = []
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, corrects = self._compute_batch_loss(batch, is_validation=True)
                batch_size = batch[0].size(0)
                running_loss += loss.item() * batch_size
                running_corrects += corrects
                total_samples += batch_size

        return running_loss / total_samples, running_corrects / total_samples

    def _compute_batch_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], is_validation: bool
    ) -> Tuple[torch.Tensor, int]:
        """Computes loss and correct predictions for a single batch.

        Args:
            batch: Tuple containing feature tensor and target tensor.
            is_validation: Boolean indicating whether to accumulate probabilities for metrics.

        Returns:
            Tuple of (loss_tensor, number_of_correct_predictions).
        """
        features, targets = batch
        features = features.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        predictions = self.model(features)

        loss = self.loss_fn(predictions, targets)
        predicted_labels = (predictions >= self.pred_target_limiar).float()
        correct_count = (predicted_labels == targets).sum().item()

        if is_validation:
            self.val_preds.append(predictions.cpu().detach().numpy())
            self.val_targets.append(targets.cpu().detach().numpy())

        return loss, correct_count
