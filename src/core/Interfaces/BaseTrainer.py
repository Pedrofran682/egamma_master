from abc import ABC, abstractmethod
import torch

from src.core.Interfaces.BaseEgammaDataset import BaseEgammaDataset


class BaseTrainer(ABC):
    """Abstract base class for model trainers.

    Attributes:
        device: PyTorch device on which tensors and models are placed.
        full_dataset: Full dataset instance covering all regions.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """PyTorch computation device."""
        pass

    @property
    @abstractmethod
    def full_dataset(self) -> BaseEgammaDataset:
        """Full dataset instance."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Executes the training workflow."""
        pass
