import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class HoldoutEvaluationResult:
    """Encapsulates model inference predictions and statistical metrics on holdout data.

    Attributes:
        x_holdout_full: Matrix of features for holdout events.
        y_holdout: Vector of true binary labels for holdout events.
        preds_05: Binary predictions using the standard 0.5 threshold.
        preds_cut: Binary predictions using the target PD threshold cut.
        probs: Continuous model output probability scores.
        global_acc: Overall accuracy at 0.5 threshold.
        signal_acc: True positive rate (TPR) at 0.5 threshold.
        bg_acc: True negative rate (TNR) at 0.5 threshold.
        fpr: False positive rate (FPR) at 0.5 threshold.
        fnr: False negative rate (FNR) at 0.5 threshold.
        cutoff_percentile: Cutoff percentile used for establishing the target PD.
        target_threshold: Model score decision threshold corresponding to target PD.
        pf: False alarm probability (PF) obtained at the target PD threshold.
    """

    x_holdout_full: np.ndarray
    y_holdout: np.ndarray
    preds_05: np.ndarray
    preds_cut: np.ndarray
    probs: np.ndarray
    global_acc: float
    signal_acc: float
    bg_acc: float
    fpr: float
    fnr: float
    cutoff_percentile: float
    target_threshold: float
    pf: float


class HoldoutEvaluator:
    """Performs inference and operating point threshold calculations on holdout data.

    Attributes:
        default_target_pd: Fallback probability of detection target (default 0.9424).
    """

    def __init__(self, default_target_pd: float = 0.9424) -> None:
        """Initializes HoldoutEvaluator with default operating point settings.

        Args:
            default_target_pd: Default target signal detection efficiency.
        """
        self.default_target_pd: float = default_target_pd

    def evaluate(
        self,
        model: nn.Module,
        data: np.ndarray,
        target: np.ndarray,
        test_indices: np.ndarray,
        ring_column_indices: np.ndarray,
        device: torch.device,
        target_pd: float | None = None,
    ) -> HoldoutEvaluationResult:
        """Performs model inference and evaluates classification metrics at fixed PD.

        Args:
            model: Trained neural network model.
            data: Regional feature matrix.
            target: Regional target labels array.
            test_indices: Indices of events designated as holdout test set.
            ring_column_indices: Column indices corresponding to calorimeter ring inputs.
            device: Active PyTorch computation device.
            target_pd: Target signal efficiency (e.g. 0.9424).

        Returns:
            HoldoutEvaluationResult containing predictions, probabilities, and rates.
        """
        eff_target = target_pd if target_pd is not None else self.default_target_pd
        model.eval()

        x_holdout_full = data[test_indices]
        y_holdout = target[test_indices]
        x_holdout_rings = x_holdout_full[:, ring_column_indices]

        with torch.no_grad():
            tensor_in = torch.from_numpy(x_holdout_rings).float().to(device)
            probs_tensor = model(tensor_in)
            probs = probs_tensor.detach().cpu().numpy().flatten()

            preds_05 = (probs > 0.5).astype(int)

            global_acc = float(np.mean(preds_05 == y_holdout))
            signal_acc = float(np.mean(preds_05[y_holdout == 1] == 1))
            bg_acc = float(np.mean(preds_05[y_holdout == 0] == 0))
            fpr = float(np.mean(preds_05[y_holdout == 0] == 1))
            fnr = float(np.mean(preds_05[y_holdout == 1] == 0))

            signal_scores = probs[y_holdout == 1]
            bg_scores = probs[y_holdout == 0]

            cutoff_percentile = (1 - eff_target) * 100
            new_threshold = float(np.percentile(signal_scores, cutoff_percentile))
            false_alarms = (bg_scores > new_threshold).sum()
            pf = float(false_alarms / len(bg_scores)) if len(bg_scores) > 0 else 0.0

            preds_cut = (probs > new_threshold).astype(int)

        return HoldoutEvaluationResult(
            x_holdout_full=x_holdout_full,
            y_holdout=y_holdout,
            preds_05=preds_05,
            preds_cut=preds_cut,
            probs=probs,
            global_acc=global_acc,
            signal_acc=signal_acc,
            bg_acc=bg_acc,
            fpr=fpr,
            fnr=fnr,
            cutoff_percentile=cutoff_percentile,
            target_threshold=new_threshold,
            pf=pf,
        )
