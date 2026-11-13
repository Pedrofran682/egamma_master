import logging
from typing import Any, Dict, Tuple
import numpy as np
import numpy.typing as npt

from src.core.Interfaces.BaseEgammaDataset import BaseEgammaDataset

log = logging.getLogger()


class EgammaNpzDatasetNoTargetOrigin(BaseEgammaDataset):
    """PyTorch Dataset loading particle physics calorimeter rings from NPZ files without MC origin filtering."""

    def filter_events(
        self, samples: Dict[str, Any]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """Filters signal (photons) and background (hadronic jets) events without MC origin constraints.

        Args:
            samples: Raw dictionary loaded from the NPZ archive.

        Returns:
            Tuple of (filtered_dataset_matrix, binary_labels_vector).
        """
        data = samples["data"]
        features = samples["feature"]
        source = samples["source"]

        mc_type_index = np.where(features == "mc_type")[0][0]
        mc_origin_index = np.where(features == "mc_origin")[0][0]

        mc_type = data[:, mc_type_index].flatten().astype(int)
        mc_origin = data[:, mc_origin_index].flatten().astype(int)

        target = np.isin(mc_type, [13, 14, 15]).astype(int)

        mask_signal = (target == 1)
        mask_jf17_data = np.char.find(source.astype(str), "perf_JF") != -1
        mask_bg = (target == 0) & (mc_origin == 42) & mask_jf17_data

        if self.config and getattr(self.config, "use_trigger_filter", False):
            log.warning("Using ph var filer. This should not be used.")
            ph_trigger_index = np.where(features == self.config.trigger_filter)[0][0]
            ph_trigger = data[:, ph_trigger_index].flatten().astype(int)
            mask_signal = mask_signal & (ph_trigger == 1)

        signal_data_full = data[mask_signal]
        bg_data_full = data[mask_bg]

        dataset_full = np.concatenate((signal_data_full, bg_data_full), axis=0)
        y_signal = np.ones((signal_data_full.shape[0], 1), dtype=np.int32)
        y_bg = np.zeros((bg_data_full.shape[0], 1), dtype=np.int32)
        y = np.vstack([y_signal, y_bg]).ravel()

        return dataset_full, y
