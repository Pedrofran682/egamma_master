import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from typing import Tuple, Dict, Any
import logging
from logging.config import fileConfig
log = logging.getLogger()


class SPCallbackPyTorch:
    def __init__(self, verbose: bool = True, 
                 save_the_best: bool = True, patience: int = 10,
                 kernel_size: int = 2, num_filters: int = 4):

        super().__init__()
        self.model = None
        self.__verbose = verbose
        self.__patience = patience
        self.__ipatience = 0
        self.__best_sp = 0.0
        self.__save_the_best = save_the_best
        self.__best_weights = None
        self.__best_epoch = 0
        
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        self.__best_fa_at_knee = None
        self.__best_pd_at_knee = None
        self.callbackMetrics = self._create_log_dict()

    def __get_partial_derivative_fa(self, fa: float, pd: float) -> float:
        c = 0.353553
        pd_fa_sqrt = np.sqrt(max(1e-9, pd * (1 - fa)))
        pd_fa_term = max(1e-9, np.sqrt(pd_fa_sqrt * (pd - fa + 1)))

        up = -(pd * (pd - fa + 1)) / (2 * pd_fa_sqrt) - pd_fa_sqrt
        down = pd_fa_term
        return c * up / down

    def __get_partial_derivative_pd(self, fa: float, pd: float) -> float:
        c = 0.353553
        pd_fa_sqrt = np.sqrt(max(1e-9, pd * (1 - fa)))
        pd_fa_term = max(1e-9, np.sqrt(pd_fa_sqrt * (pd - fa + 1)))

        up = ((1 - fa) * (pd - fa + 1)) / (2 * pd_fa_sqrt) + pd_fa_sqrt
        down = pd_fa_term
        return c * up / down
    
    def _create_log_dict(self):
        return {
            'max_sp_val': [],
            'max_sp_fa_val': [],
            'max_sp_pd_val': [],
            'max_sp_partial_derivative_fa_val': [],
            'max_sp_partial_derivative_pd_val': [],
            'fa': [],
            'pd': [],
            'auc_score': []
        }

    def on_epoch_end(self,
                     model : torch.nn.Module,
                     epoch: int, 
                     y_true: np.ndarray, 
                     y_pred: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        self.model = model
        if not isinstance(y_true, np.ndarray):
            y_true = np.concatenate(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.concatenate(y_pred)

        y_pred = y_pred.ravel()
        stop_training = False
        
        try:
            false_positive_rate, true_positive_rates, _ = roc_curve(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_pred)
            sp_values = np.sqrt(np.sqrt(true_positive_rates * (1 - false_positive_rate)) * (0.5 * (true_positive_rates + (1 - false_positive_rate))))
        except ValueError as e:
            log.warning(f"Error calculating roc_curve in epoch {epoch}: {e}. Skipping knee calculation.")
            sp_values = np.array([])

        if sp_values.size == 0 or np.all(np.isnan(sp_values)):
            if self.__verbose:
                log.warning(f"sp_values is empty or contains NaNs in epoch {epoch}. Skipping knee calculation.")
            self.callbackMetrics["max_sp_val"] = 0.0
            self.callbackMetrics["max_sp_fa_val"] = 0.0
            self.callbackMetrics["max_sp_pd_val"] = 0.0
            self.callbackMetrics["max_sp_partial_derivative_fa_val"] = 0.0
            self.callbackMetrics["max_sp_partial_derivative_pd_val"] = 0.0
            self.callbackMetrics["fa"] = []
            self.callbackMetrics["pd"] = []
            self.callbackMetrics["auc_score"] = 0.0
            return stop_training, self.callbackMetrics

        knee = np.argmax(sp_values)
        current_sp = sp_values[knee]
        false_positive_rate_knee = false_positive_rate[knee]
        true_positive_rates_knee = true_positive_rates[knee]
        partial_fa = self.__get_partial_derivative_fa(false_positive_rate_knee, true_positive_rates_knee)
        partial_pd = self.__get_partial_derivative_pd(false_positive_rate_knee, true_positive_rates_knee)

        self.callbackMetrics["max_sp_val"].append(current_sp)
        self.callbackMetrics["max_sp_fa_val"].append(false_positive_rate_knee)
        self.callbackMetrics["max_sp_pd_val"].append(true_positive_rates_knee)
        self.callbackMetrics["max_sp_partial_derivative_fa_val"].append(partial_fa)
        self.callbackMetrics["max_sp_partial_derivative_pd_val"].append(partial_pd)

        if self.__verbose:
            log_message = (
                f"Epoch {epoch}: - val_sp: {current_sp:.4f} (fa:{false_positive_rate_knee:.4f}, pd:{true_positive_rates_knee:.4f}), "
                f"patience: {self.__ipatience}/{self.__patience}, dSP/dFA: {partial_fa:.4f}, dSP/dPD: {partial_pd:.4f}"
            )
            log.info(log_message)

        if current_sp > self.__best_sp:
            self.__best_sp = current_sp
            self.__ipatience = 0
            self.callbackMetrics["fa"] = false_positive_rate
            self.callbackMetrics["pd"] = true_positive_rates
            self.callbackMetrics["auc_score"] = auc_score

            if self.__save_the_best:
                self.__best_weights = self.model.state_dict()
                self.__best_epoch = epoch
                self.__best_fa_at_knee = false_positive_rate_knee
                self.__best_pd_at_knee = true_positive_rates_knee
                if self.__verbose:
                    log.info(f"Epoch {epoch}: val_sp improved to {current_sp:.4f}. Saving model.")
        else:
            self.__ipatience += 1

        if self.__ipatience > self.__patience:
            if self.__verbose:
                log.info(f"Early stopping triggered at epoch {epoch}. Best SP was {self.__best_sp:.4f} at epoch {self.__best_epoch}.")
            stop_training = True

        return stop_training, self.callbackMetrics

    def get_best_model_weights(self) -> Dict[str, Any]:
        return self.__best_weights

    def get_best_sp_value(self) -> float:
        return self.__best_sp

    def get_best_fa_at_knee(self) -> float:
        return self.__best_fa_at_knee

    def get_best_pd_at_knee(self) -> float:
        return self.__best_pd_at_knee
    
    def reset_state(self):
        self.__ipatience = 0
        self.__best_sp = 0.0
        self.__best_epoch = 0
        self.__best_fa_at_knee = None
        self.__best_pd_at_knee = None