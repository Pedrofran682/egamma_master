import logging
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_class_weight, get_instance

log = logging.getLogger()


class TrainingFactory:
    """Factory responsible for constructing PyTorch models, optimizers, losses, and DataLoaders.

    Attributes:
        config: Training configuration settings.
        device: Target execution torch device (CPU or CUDA).
        use_cuda: Boolean indicating whether CUDA is active.
    """

    def __init__(self, config: NeuralRingerTrainerConfiguration, device: torch.device) -> None:
        """Initializes TrainingFactory with configuration and hardware device.

        Args:
            config: Trainer configuration instance.
            device: Active PyTorch computation device.
        """
        self.config: NeuralRingerTrainerConfiguration = config
        self.device: torch.device = device
        self.use_cuda: bool = device.type == "cuda"

    def create_model(self) -> nn.Module:
        """Instantiates the PyTorch model and transfers it to the execution device.

        Returns:
            Configured and device-mapped nn.Module instance.
        """
        model = get_instance(self.config.model)
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(self.device)
        return model

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Instantiates the optimizer bound to model parameters.

        Args:
            model: PyTorch model whose parameters will be optimized.

        Returns:
            Configured torch.optim.Optimizer instance.
        """
        self.config.optimizer_function.parameters["params"] = model.parameters()
        return get_instance(self.config.optimizer_function)

    def create_loss_function(self) -> nn.Module:
        """Instantiates the loss criterion configured for the run.

        Returns:
            Configured loss function module.
        """
        return get_instance(self.config.loss_function)

    def create_cross_validation(self) -> Any:
        """Instantiates the K-Fold cross validation splitter.

        Returns:
            Cross-validation splitter instance (e.g. StratifiedGroupKFold).
        """
        return get_instance(self.config.kFold)

    def create_dataloader(self, data: np.ndarray, labels: np.ndarray) -> DataLoader:
        """Constructs a PyTorch DataLoader with optional class weighting and GPU pinning.

        Args:
            data: Input numpy feature array.
            labels: Target labels array.

        Returns:
            Configured PyTorch DataLoader ready for batch iteration.
        """
        features_tensor = torch.from_numpy(data).float()
        labels_tensor = torch.from_numpy(labels).float().view(-1, 1)

        dataset = TensorDataset(features_tensor, labels_tensor)
        batch_size = self.config.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        sampler = None
        shuffle = True
        if self.config.balance_data:
            sampler = get_class_weight(labels.astype(int))
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.use_cuda,
            shuffle=shuffle,
            sampler=sampler,
        )
