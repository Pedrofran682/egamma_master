import logging
import os
import pathlib
import re
from typing import List
import numpy as np
import pandas as pd
import yaml

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
            output_dir: Optional destination directory for plots (defaults to Plots/<yaml_name>).
        """
        self.config_path: pathlib.Path = pathlib.Path(config_path)
        self.config_instance: NeuralRingerTrainerConfiguration = self._load_config(config_path)
        self.data_path: pathlib.Path = pathlib.Path(data_path)
        if output_dir is not None:
            self.plot_dir: pathlib.Path = pathlib.Path(output_dir)
        else:
            self.plot_dir: pathlib.Path = pathlib.Path("Plots") / self.config_path.stem
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
        """Executes full validation workflow across all kinematic regions."""
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
            all_paths = np.array([item for item in self.trainer.full_dataset.file_paths])
            matched_indices = np.where(np.char.find(all_paths, file_name_pattern) != -1)[0]

            if len(matched_indices) == 0:
                log.warning(f"File pattern '{file_name_pattern}' not found. Skipping region...")
                continue

            target_index = matched_indices[0]
            data, target, _ = self.trainer.full_dataset[target_index]
            test_indices, _ = self.trainer.generate_folds_with_holdout(data, target)

            best_model, best_rep, mean_sp, std_sp = self.aggregator.get_best_model_details(df)
            self.model.load_state_dict(best_model["best_weights"])

            context = RegionPlotContext(
                iet=iet,
                ieta=ieta,
                output_dir=self.plot_dir,
                data=data,
                target=target,
                test_indices=test_indices,
                trainer=self.trainer,
                model=self.model,
            )

            self.roc_plotter.plot(
                context,
                best_model_details=best_model["history"]["callbackMetrics"],
            )

            self.profile_plotter.plot(
                context,
                x_rings=context.x_holdout_rings,
                plot_name="RingsMeanProfiles_NeuralRinger",
            )

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
