from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd

from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration


class BaseHyperparameterTuner(ABC):
    """Abstract base class for hyperparameter tuning orchestrators."""

    @abstractmethod
    def generate_trials(self) -> List[Tuple[str, NeuralRingerTrainerConfiguration]]:
        """Generates list of trial tags and associated trainer configurations.

        Returns:
            List of tuples containing unique trial identifier and trainer config.
        """
        pass

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Executes hyperparameter trials and aggregates performance metrics.

        Returns:
            DataFrame containing evaluated trials sorted by primary optimization metric.
        """
        pass
