import logging
import os
import pathlib
import re
from typing import List
import numpy as np
import pandas as pd
import yaml

from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.MetricPlotter import RocPlotter
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.EfficiencyPlotter import EfficiencyPlotter
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluationResult, HoldoutEvaluator
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration

log = logging.getLogger()


class ModelValidator:
    """Orchestrator for model validation, holdout benchmark evaluation, and visual plotting.

    Attributes:
        config_instance: NeuralRingerTrainerConfiguration object.
        data_path: Path to the directory holding model run pickles.
        manifest_path: Path to the expected split manifest JSON file.
        manifest: SplitManifest manager instance for loading deterministic partitions.
        plot_dir: Directory where figures and plots will be placed.
        reference_eff_df: Optional reference efficiency benchmark DataFrame.
        trainer: NeuralRingerTrainer instance used for dataset indexing and device context.
        model: PyTorch model loaded with optimal checkpoint weights.
        aggregator: ResultAggregator scanning fold and repeat performance.
        evaluator: HoldoutEvaluator computing statistical performance.
        efficiency_plotter: EfficiencyPlotter rendering response curves.
        profile_plotter: ProfileMeanEnergyPlotter rendering calorimeter rings.
        roc_plotter: RocPlotter rendering ROC curves.
    """

    def __init__(
        self,
        config_path: str,
        data_path: str,
        efficiencies_csv: str = "efficiencies_by_region.csv",
        default_target_pd: float = 0.9424,
        output_dir: str | pathlib.Path | None = None,
    ) -> None:
        """Initializes the ModelValidator pipeline.

        Args:
            config_path: YAML file path containing run configuration.
            data_path: Directory path holding result pickle archives.
            efficiencies_csv: Path to reference operating point CSV table.
            default_target_pd: Fallback target signal efficiency (default 0.9424).
            output_dir: Optional destination directory for plots (defaults to Plots/<results_folder_name>).
        """
        self.config_path: pathlib.Path = pathlib.Path(config_path)
        self.config_instance: NeuralRingerTrainerConfiguration = self._load_config(config_path)
        self.data_path: pathlib.Path = pathlib.Path(data_path)
        self.manifest_path: pathlib.Path = self.data_path / "split_manifest.json"
        self.manifest: SplitManifest = SplitManifest(str(self.manifest_path))
        if output_dir is not None:
            self.plot_dir: pathlib.Path = pathlib.Path(output_dir)
        else:
            target_name = (
                self.data_path.parent.name
                if self.data_path.is_file()
                or self.data_path.suffix in [".pkl", ".npz", ".pt"]
                else (self.data_path.name or "default")
            )
            self.plot_dir: pathlib.Path = pathlib.Path("Plots") / target_name
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.reference_eff_df: pd.DataFrame | None = self._load_reference_efficiencies(efficiencies_csv)

        self.trainer: NeuralRingerTrainer = NeuralRingerTrainer(self.config_instance)
        self.model = self.trainer.factory.create_model()
        self.aggregator: ResultAggregator = ResultAggregator(self.data_path)
        self.evaluator: HoldoutEvaluator = HoldoutEvaluator(default_target_pd=default_target_pd)
        self.efficiency_plotter: EfficiencyPlotter = EfficiencyPlotter()
        self.profile_plotter: ProfileMeanEnergyPlotter = ProfileMeanEnergyPlotter()
        self.roc_plotter: RocPlotter = RocPlotter()

    @staticmethod
    def _load_config(config_path: str) -> NeuralRingerTrainerConfiguration:
        """Loads and parses a validated YAML configuration.

        Args:
            config_path: Path to configuration YAML.

        Returns:
            Validated NeuralRingerTrainerConfiguration instance.
        """
        with open(config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            return NeuralRingerTrainerConfiguration.model_validate(yaml_data)

    @staticmethod
    def _load_reference_efficiencies(path: str) -> pd.DataFrame | None:
        """Loads reference benchmark efficiencies from CSV if available.

        Args:
            path: CSV file path.

        Returns:
            Parsed DataFrame or None if file does not exist.
        """
        if os.path.exists(path):
            log.info(f"Loading reference efficiencies from {path}")
            return pd.read_csv(path)
        return None

    def _get_target_pd(self, iet: int, ieta: int) -> float:
        """Retrieves region-specific target PD efficiency from reference table.

        Args:
            iet: Transverse energy bin index.
            ieta: Pseudorapidity bin index.

        Returns:
            Target efficiency float value.
        """
        if self.reference_eff_df is not None:
            ref_row = self.reference_eff_df[
                (self.reference_eff_df["et_bin"] == iet)
                & (self.reference_eff_df["eta_bin"] == ieta)
            ]
            if not ref_row.empty:
                return float(ref_row["eff_sig_loose"].values[0])
        return self.evaluator.default_target_pd

    def run(self) -> None:
        """Executes full validation workflow across all kinematic regions.

        Raises:
            FileNotFoundError: If the split manifest file does not exist in data_path.
            KeyError: If a kinematic region key is missing from the split manifest.
        """
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Split manifest not found at {self.manifest_path}. Validation requires persisted splits."
            )
        self.manifest.load()

        grouped_dfs = self.aggregator.group_and_concat()
        all_results: List[HoldoutEvaluationResult] = []

        for region_name, df in grouped_dfs.items():
            if region_name == "undefined_region":
                continue

            match = re.search(r"iet(\d+)\.ieta(\d+)", region_name)
            iet = int(match.group(1)) if match else 1
            ieta = int(match.group(2)) if match else 1
            target_pd = self._get_target_pd(iet, ieta)

            file_name_pattern = f"consolidated.et{iet}.eta{ieta}.npz"
            all_paths = np.array(
                [str(item) for item in self.trainer.full_dataset.file_paths], dtype=str
            )
            matched_indices = np.where(np.char.find(all_paths, file_name_pattern) != -1)[0]

            if len(matched_indices) == 0:
                log.warning(f"File pattern '{file_name_pattern}' not found. Skipping region...")
                continue

            region_key = f"et_{iet}_eta_{ieta}"
            test_indices, _ = self.manifest.get_region_splits(region_key)

            target_index = matched_indices[0]
            data, target, data_path = self.trainer.full_dataset[target_index]
            log.info(f"Using {data_path} for region {region_name}")

            best_model, best_rep, mean_sp, std_sp = self.aggregator.get_best_model_details(df)
            self.model.load_state_dict(best_model["best_weights"])

            result = self.evaluator.evaluate(
                model=self.model,
                data=data,
                target=target,
                test_indices=test_indices,
                ring_column_indices=self.trainer.full_dataset.ring_column_indices,
                device=self.trainer.device,
                target_pd=target_pd,
            )

            all_results.append(result)

        validation_dir = self.plot_dir / "Validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        self.efficiency_plotter.generate_global_plots(
            all_results,
            validation_dir,
            self.config_instance.config_name,
        )
