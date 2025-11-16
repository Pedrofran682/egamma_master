import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
from src.utils import norm1
import logging
log = logging.getLogger()
from functools import lru_cache


class EgammaNpzDataset(Dataset):
    def __init__(self, file_paths, percentage=1.0):
        log.info("Creating Dataset...")
        self.file_paths: list[str] = file_paths
        self.index_last_ring = 101
        self.percentage: int = percentage
        self.percentage_dim: int = None
        self.indexes = self._get_rings_index(self.percentage)
        self.percentage_dim = len(self.indexes)
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        log.info(f"Getting dataset index {idx}: {self.file_paths[idx]}")
        self.percentage_dim = None
        features, labels = None, None
        with np.load(self.file_paths[idx]) as samples:
            features, labels = self._create_rings(samples['data'],
                                                  samples['target'])
        return features, labels, self.file_paths[idx]

    def _create_rings(self, data: np.ndarray,
                      target: np.ndarray) -> Tuple[Tuple[np.ndarray,np.ndarray],
                                                Tuple[np.ndarray,np.ndarray]]:
        
        signal_data = data[np.where(target == 1)]
        bg_data = data[np.where(target == 0)]

        signal_data = signal_data[:, 1: self.index_last_ring]
        bg_data = bg_data[:, 1: self.index_last_ring]
        signal_data = signal_data[:, self.indexes]
        bg_data = bg_data[:, self.indexes]

        dataset = np.concatenate(
            (
                norm1(signal_data),
                norm1(bg_data)
            ),
            axis=0
        )
        y_sinal = np.ones((signal_data.shape[0], 1), dtype=np.float32)
        y_background = np.zeros((bg_data.shape[0], 1), dtype=np.float32)
        y = np.vstack([y_sinal, y_background]).ravel()
        return dataset, y

    @lru_cache(maxsize=None)
    def _get_rings_index(self, percentage: float) -> list[int]:
        PreSampler_index = [index for index in range(0, 8)]
        EM1_index = [index for index in range(8, 72)]
        EM2_index = [index for index in range(72, 80)]
        EM3_index = [index for index in range(80, 88)]
        TileCal_index = [index for index in range(88, 100)]
        
        PreSampler_index_cap = round(len(PreSampler_index) * percentage)
        EM1_index_cap = round(len(EM1_index) * percentage)
        EM2_index_cap = round(len(EM2_index) * percentage)
        EM3_index_cap = round(len(EM3_index) * percentage)
        TileCal_index_cap = round(len(TileCal_index) * percentage)
    
        PreSampler_index = PreSampler_index[: PreSampler_index_cap]
        EM1_index = EM1_index[: EM1_index_cap]
        EM2_index = EM2_index[: EM2_index_cap]
        EM3_index = EM3_index[: EM3_index_cap]
        TileCal_index = TileCal_index[: TileCal_index_cap]
    
        return PreSampler_index + EM1_index + EM2_index + EM3_index + TileCal_index

    def get_model_dim(self) -> int:
        if self.percentage_dim != None:
            return self.percentage_dim
        else:
            raise Exception("Dimension was not set.")