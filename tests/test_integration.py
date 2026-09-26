import pathlib
from typing import Any, Dict

import numpy as np
import pytest
import torch

from src.Models.Models import get_model
from src.Parser.DynamicConfiguration import DynamicConfiguration
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.ResultAggregator import ResultAggregator


class TestIntegrationPipeline:
    """Integration test suite executing the complete training, persistence, and aggregation lifecycle."""

    @pytest.fixture
    def synthetic_data_dir(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Generates a synthetic NPZ dataset file adhering to calorimeter ringer specifications."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        n_signal = 50
        n_background = 50
        n_total = n_signal + n_background

        feature_names = ["mc_type", "mc_origin"] + [
            f"trig_L2_calo_rings_{i}" for i in range(100)
        ]

        data_matrix = np.zeros((n_total, len(feature_names)), dtype=np.float32)

        # Signal events (mc_type=14, mc_origin=3)
        data_matrix[:n_signal, 0] = 14
        data_matrix[:n_signal, 1] = 3
        data_matrix[:n_signal, 2:] = np.random.uniform(0.5, 2.0, size=(n_signal, 100))

        # Background events (mc_type=0, mc_origin=42)
        data_matrix[n_signal:, 0] = 0
        data_matrix[n_signal:, 1] = 42
        data_matrix[n_signal:, 2:] = np.random.uniform(0.0, 1.0, size=(n_background, 100))

        source_array = np.array(["mc23_13TeV.perf_JF"] * n_total)

        file_name = "mc23_13TeV.sample.et1.eta1.npz"
        np.savez(
            data_dir / file_name,
            data=data_matrix,
            feature=np.array(feature_names),
            source=source_array,
        )
        return data_dir

    @pytest.fixture
    def integration_config(
        self, synthetic_data_dir: pathlib.Path, tmp_path: pathlib.Path
    ) -> NeuralRingerTrainerConfiguration:
        """Builds a configuration object for fast end-to-end integration testing."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        config_dict: Dict[str, Any] = {
            "config_name": "IntegrationPipelineRun",
            "batch_size": 16,
            "epochs": 2,
            "et_range_idx": [1],
            "eta_range_idx": [1],
            "n_initializations": 1,
            "pred_target_limiar": 0.5,
            "use_trigger_filter": False,
            "trigger_filter": "",
            "num_workers": 0,
            "debug": False,
            "results_folder_path": str(results_dir),
            "balance_data": True,
            "loss_function": {
                "module": "torch.nn",
                "object_name": "BCELoss",
                "parameters": {},
            },
            "optimizer_function": {
                "module": "torch.optim",
                "object_name": "Adam",
                "parameters": {"lr": 0.01},
            },
            "model": {
                "module": "src.Models.egamma.ModelV6",
                "object_name": "ModelV6",
                "parameters": {"input_dim": 50},
            },
            "dataset": {
                "module": "src.core.Datasets.EgammaNpzDataset",
                "object_name": "EgammaNpzDataset",
                "parameters": {
                    "drive_path": str(synthetic_data_dir),
                    "startswith": "mc23_13TeV",
                    "endswith": ".npz",
                    "percentage": 0.5,
                },
            },
            "kFold": {
                "module": "sklearn.model_selection",
                "object_name": "StratifiedKFold",
                "parameters": {"n_splits": 3, "shuffle": True, "random_state": 42},
            },
        }
        return NeuralRingerTrainerConfiguration.model_validate(config_dict)

    def test_full_training_to_aggregation_pipeline(
        self,
        integration_config: NeuralRingerTrainerConfiguration,
        tmp_path: pathlib.Path,
    ) -> None:
        """Executes full pipeline: train folds, record metrics, persist manifest, aggregate results."""
        results_path = pathlib.Path(str(integration_config.results_folder_path))

        # 1. Run Trainer
        trainer = NeuralRingerTrainer(integration_config)
        trainer.run()

        # 2. Verify split manifest creation and persistence
        manifest_file = results_path / "split_manifest.json"
        assert manifest_file.exists()

        # 3. Verify fold result pickle files exist
        pkl_files = list(results_path.glob("repeat*.pkl"))
        assert len(pkl_files) == 2

        # 4. Verify ResultAggregator groups and extracts top performance
        aggregator = ResultAggregator(results_path)
        grouped_results = aggregator.group_and_concat()
        assert "iet1.ieta1" in grouped_results

        df_region = grouped_results["iet1.ieta1"]
        assert len(df_region) == 2  # 2 folds

        best_model_details, best_rep, mean_sp, std_sp = (
            aggregator.get_best_model_details(df_region)
        )
        assert best_rep == 1
        assert "best_weights" in best_model_details
        assert best_model_details["best_weights"] is not None
        assert mean_sp > 0.0

        # 5. Verify ModelRegistry can reconstruct best model with loaded weights
        model = get_model("ModelV6", input_dim=50)
        model.load_state_dict(best_model_details["best_weights"])
        model.eval()

        dummy_input = torch.randn(4, 50)
        with torch.no_grad():
            preds = model(dummy_input)
        assert preds.shape == (4, 1)
