import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from src.core.Interfaces.BaseEvaluator import BaseEvaluator

log = logging.getLogger()


@dataclass
class HoldoutEvaluationResult:
    """Encapsulates model inference predictions and statistical metrics on holdout data.

    Attributes:
        holdout_features_full: Matrix of features for holdout events.
        holdout_labels: Vector of true binary labels for holdout events.
        predictions_default_threshold: Binary predictions using the standard 0.5 threshold.
        predictions_calibrated_cut: Binary predictions using the target PD threshold cut.
        predicted_probabilities: Continuous model output probability scores.
        accuracy_default_threshold: Overall accuracy at 0.5 threshold.
        signal_efficiency_default_threshold: True positive rate (TPR) at 0.5 threshold.
        background_rejection_default_threshold: True negative rate (TNR) at 0.5 threshold.
        false_positive_rate: False positive rate (FPR) at 0.5 threshold.
        false_negative_rate: False negative rate (FNR) at 0.5 threshold.
        signal_cutoff_percentile: Cutoff percentile used for establishing the target PD.
        calibrated_decision_threshold: Model score decision threshold corresponding to target PD.
        false_alarm_probability: False alarm probability obtained at the target PD threshold.
    """

    holdout_features_full: np.ndarray
    holdout_labels: np.ndarray
    predictions_default_threshold: np.ndarray
    predictions_calibrated_cut: np.ndarray
    predicted_probabilities: np.ndarray
    accuracy_default_threshold: float
    signal_efficiency_default_threshold: float
    background_rejection_default_threshold: float
    false_positive_rate: float
    false_negative_rate: float
    signal_cutoff_percentile: float
    calibrated_decision_threshold: float
    false_alarm_probability: float

    @property
    def x_holdout_full(self) -> np.ndarray:
        """Matrix of features for holdout events (backward-compatible alias)."""
        return self.holdout_features_full

    @property
    def y_holdout(self) -> np.ndarray:
        """Vector of true binary labels for holdout events (backward-compatible alias)."""
        return self.holdout_labels

    @property
    def preds_05(self) -> np.ndarray:
        """Binary predictions using standard 0.5 threshold (backward-compatible alias)."""
        return self.predictions_default_threshold

    @property
    def preds_cut(self) -> np.ndarray:
        """Binary predictions using target PD threshold cut (backward-compatible alias)."""
        return self.predictions_calibrated_cut

    @property
    def probs(self) -> np.ndarray:
        """Continuous model output probability scores (backward-compatible alias)."""
        return self.predicted_probabilities

    @property
    def global_acc(self) -> float:
        """Overall accuracy at 0.5 threshold (backward-compatible alias)."""
        return self.accuracy_default_threshold

    @property
    def signal_acc(self) -> float:
        """True positive rate at 0.5 threshold (backward-compatible alias)."""
        return self.signal_efficiency_default_threshold

    @property
    def bg_acc(self) -> float:
        """True negative rate at 0.5 threshold (backward-compatible alias)."""
        return self.background_rejection_default_threshold

    @property
    def fpr(self) -> float:
        """False positive rate at 0.5 threshold (backward-compatible alias)."""
        return self.false_positive_rate

    @property
    def fnr(self) -> float:
        """False negative rate at 0.5 threshold (backward-compatible alias)."""
        return self.false_negative_rate

    @property
    def cutoff_percentile(self) -> float:
        """Cutoff percentile for establishing target PD (backward-compatible alias)."""
        return self.signal_cutoff_percentile

    @property
    def target_threshold(self) -> float:
        """Decision threshold corresponding to target PD (backward-compatible alias)."""
        return self.calibrated_decision_threshold

    @property
    def pf(self) -> float:
        """False alarm probability at target PD threshold (backward-compatible alias)."""
        return self.false_alarm_probability


class HoldoutEvaluator(BaseEvaluator):
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
        target_signal_efficiency = target_pd if target_pd is not None else self.default_target_pd
        model.eval()

        if ring_column_indices is None:
            raise ValueError("ring_column_indices cannot be None during holdout evaluation.")

        holdout_features_full = data[test_indices]
        holdout_labels = target[test_indices]
        holdout_ring_features = holdout_features_full[:, ring_column_indices]

        with torch.no_grad():
            ring_features_tensor = torch.from_numpy(holdout_ring_features).float().to(device)
            probabilities_tensor = model(ring_features_tensor)
            predicted_probabilities = probabilities_tensor.detach().cpu().numpy().flatten()

            predictions_default_threshold = (predicted_probabilities > 0.5).astype(int)

            accuracy_default_threshold = float(np.mean(predictions_default_threshold == holdout_labels))
            signal_efficiency_default_threshold = float(
                np.mean(predictions_default_threshold[holdout_labels == 1] == 1)
            )
            background_rejection_default_threshold = float(
                np.mean(predictions_default_threshold[holdout_labels == 0] == 0)
            )
            false_positive_rate = float(np.mean(predictions_default_threshold[holdout_labels == 0] == 1))
            false_negative_rate = float(np.mean(predictions_default_threshold[holdout_labels == 1] == 0))

            signal_probability_scores = predicted_probabilities[holdout_labels == 1]
            background_probability_scores = predicted_probabilities[holdout_labels == 0]

            signal_cutoff_percentile = (1 - target_signal_efficiency) * 100
            calibrated_decision_threshold = float(
                np.percentile(signal_probability_scores, signal_cutoff_percentile)
            )
            false_alarm_count = int((background_probability_scores > calibrated_decision_threshold).sum())
            false_alarm_probability = (
                float(false_alarm_count / len(background_probability_scores))
                if len(background_probability_scores) > 0
                else 0.0
            )

            predictions_calibrated_cut = (predicted_probabilities > calibrated_decision_threshold).astype(int)

        return HoldoutEvaluationResult(
            holdout_features_full=holdout_features_full,
            holdout_labels=holdout_labels,
            predictions_default_threshold=predictions_default_threshold,
            predictions_calibrated_cut=predictions_calibrated_cut,
            predicted_probabilities=predicted_probabilities,
            accuracy_default_threshold=accuracy_default_threshold,
            signal_efficiency_default_threshold=signal_efficiency_default_threshold,
            background_rejection_default_threshold=background_rejection_default_threshold,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            signal_cutoff_percentile=signal_cutoff_percentile,
            calibrated_decision_threshold=calibrated_decision_threshold,
            false_alarm_probability=false_alarm_probability,
        )
