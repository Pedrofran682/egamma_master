import pathlib
import unittest
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.core.Interfaces import (
    BaseEvaluator,
    BaseMetricPlotter,
    BasePlotter,
    BaseResultAggregator,
    BaseTrainer,
)
from src.core.Plotting.Context import MetricPlotContext, RegionPlotContext
from src.core.Plotting.PlotManager import PlotManager
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration


class TestInterfaces(unittest.TestCase):
    def test_base_trainer_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            BaseTrainer()

    def test_base_trainer_subclass_instantiation(self) -> None:
        class ConcreteTrainer(BaseTrainer):
            @property
            def device(self) -> torch.device:
                return torch.device("cpu")

            @property
            def full_dataset(self) -> Any:
                return None

            def run(self) -> None:
                pass

        trainer = ConcreteTrainer()
        self.assertEqual(trainer.device.type, "cpu")
        self.assertIsNone(trainer.full_dataset)

    def test_base_result_aggregator_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            BaseResultAggregator()

    def test_base_result_aggregator_subclass_instantiation(self) -> None:
        class ConcreteAggregator(BaseResultAggregator):
            def group_and_concat(self) -> Dict[str, pd.DataFrame]:
                return {}

            def has_region(self, et: int, eta: int) -> bool:
                return True

            def get_best_model_for_region(
                self, et: int, eta: int
            ) -> Tuple[Dict[str, Any], int, float, float]:
                return ({}, 0, 0.0, 0.0)

        aggregator = ConcreteAggregator()
        self.assertTrue(aggregator.has_region(0, 0))

    def test_base_evaluator_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            BaseEvaluator()

    def test_base_evaluator_subclass_instantiation(self) -> None:
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(
                self,
                model: nn.Module,
                data: np.ndarray,
                target: np.ndarray,
                test_indices: np.ndarray,
                ring_column_indices: np.ndarray,
                device: torch.device,
                target_pd: float | None = None,
            ) -> Any:
                return "evaluated"

        evaluator = ConcreteEvaluator()
        dummy_model = nn.Linear(1, 1)
        res = evaluator.evaluate(
            dummy_model,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            torch.device("cpu"),
        )
        self.assertEqual(res, "evaluated")

    def test_base_metric_plotter_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            BaseMetricPlotter("test")

    def test_base_metric_plotter_polymorphic_dispatch(self) -> None:
        dispatched_payloads = []

        class CustomPlotter(BaseMetricPlotter):
            def __init__(self) -> None:
                super().__init__(name="CustomPlotter")

            def plot_metric(self, context: MetricPlotContext) -> Any:
                dispatched_payloads.append(context)

            def plot(self, context: RegionPlotContext, **kwargs: Any) -> Any:
                pass

        manager = PlotManager(include_default_plotters=False)
        custom_plotter = CustomPlotter()
        manager.register_plotter(custom_plotter)

        df = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "reapet": 1,
                    "best_sp_value": 0.90,
                    "history": {"train_loss": [0.1]},
                }
            ]
        )
        success = manager.run_metrics_for_region(df, output_dir="dummy", iet=1, ieta=2)
        self.assertTrue(success)
        self.assertEqual(len(dispatched_payloads), 1)
        self.assertEqual(dispatched_payloads[0].region_context.iet, 1)
        self.assertEqual(dispatched_payloads[0].region_context.ieta, 2)
        self.assertEqual(dispatched_payloads[0].history_data, {"train_loss": [0.1]})

    def test_configuration_is_region_allowed(self) -> None:
        raw_config = {
            "batch_size": 32,
            "config_name": "TestConfig",
            "epochs": 10,
            "et_range_idx": [1, 2],
            "eta_range_idx": [3, 4],
            "n_initializations": 1,
            "pred_target_limiar": 0.5,
            "loss_function": {"module": "torch.nn", "object_name": "BCEWithLogitsLoss"},
            "optimizer_function": {"module": "torch.optim", "object_name": "Adam"},
            "model": {"module": "src.Models.egamma.ModelV1", "object_name": "ModelV1"},
            "dataset": {"module": "src.core.Datasets.EgammaNpzDataset", "object_name": "EgammaNpzDataset"},
            "kFold": {"module": "sklearn.model_selection", "object_name": "KFold"},
        }
        cfg = NeuralRingerTrainerConfiguration.model_validate(raw_config)
        self.assertTrue(cfg.is_region_allowed(1, 3))
        self.assertTrue(cfg.is_region_allowed(2, 4))
        self.assertFalse(cfg.is_region_allowed(0, 3))
        self.assertFalse(cfg.is_region_allowed(1, 5))

    def test_base_egamma_dataset_cannot_be_instantiated_directly(self) -> None:
        from src.core.Interfaces import BaseEgammaDataset

        with self.assertRaises(TypeError):
            BaseEgammaDataset(drive_path="dummy")

    def test_datasets_inherit_from_base_egamma_dataset(self) -> None:
        from src.core.Datasets.EgammaNpzDataset import EgammaNpzDataset
        from src.core.Datasets.EgammaNpzDatasetNoTargetOrigin import EgammaNpzDatasetNoTargetOrigin
        from src.core.Interfaces import BaseEgammaDataset

        self.assertTrue(issubclass(EgammaNpzDataset, BaseEgammaDataset))
        self.assertTrue(issubclass(EgammaNpzDatasetNoTargetOrigin, BaseEgammaDataset))


if __name__ == "__main__":
    unittest.main()
