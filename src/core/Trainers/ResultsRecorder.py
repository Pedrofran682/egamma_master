import os
from typing import Any, Dict, List
import pandas as pd
from src.utils import get_results_file_name


class ResultsRecorder:
    """Collects training run metrics, model checkpoints, and exports them to disk.

    Attributes:
        records: In-memory list of fold result dictionaries.
    """

    def __init__(self) -> None:
        """Initializes an empty ResultsRecorder."""
        self.records: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Clears accumulated records for the next fold iteration."""
        self.records = []

    def record(
        self,
        repeat: int,
        file_path: str,
        fold: int,
        best_sp: float,
        best_fa: float,
        best_pd: float,
        best_weights: Any,
        history: Dict[str, Any],
    ) -> None:
        """Appends metrics and model checkpoint state for a completed fold run.

        Args:
            repeat: Initialization / repeat index.
            file_path: Dataset source path.
            fold: Cross-validation fold index.
            best_sp: Best SP index achieved.
            best_fa: False alarm rate at optimal SP knee point.
            best_pd: Probability of detection at optimal SP knee point.
            best_weights: Best model state_dict weights.
            history: Training history dictionary containing loss/acc progressions.
        """
        self.records.append(
            {
                "reapet": repeat,
                "file_path": file_path,
                "fold": fold,
                "best_sp_value": best_sp,
                "best_fa_value": best_fa,
                "best_pd_value": best_pd,
                "best_weights": best_weights,
                "history": history,
            }
        )

    def save(
        self, folder_path: str, et: int, eta: int, repeat: int, fold_idx: int
    ) -> str:
        """Serializes current fold records into a pickled pandas DataFrame.

        Args:
            folder_path: Output directory path.
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.
            repeat: Repetition index.
            fold_idx: Fold index.

        Returns:
            Absolute file path where the pickle file was saved.
        """
        file_name = get_results_file_name(et, eta, repeat, fold_idx)
        destination_path = os.path.join(folder_path, file_name)
        pd.DataFrame(self.records).to_pickle(destination_path)
        return destination_path
