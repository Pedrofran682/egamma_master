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
from src.utils import get_et_eta

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

        validation_dir = self.plot_dir / "Validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        et_filter = getattr(self.config_instance, "et_range_idx", None)
        eta_filter = getattr(self.config_instance, "eta_range_idx", None)
        all_results: List[HoldoutEvaluationResult] = []

        for index in range(len(self.trainer.full_dataset)):
            data_path = self.trainer.full_dataset.file_paths[index]
            iet, ieta = get_et_eta(data_path)

            if isinstance(et_filter, (list, tuple, set)) and iet not in et_filter:
                continue
            if isinstance(eta_filter, (list, tuple, set)) and ieta not in eta_filter:
                continue

            if hasattr(self.aggregator, "has_region") and not self.aggregator.has_region(iet, ieta):
                log.warning(f"No result records found for region iet{iet}.ieta{ieta}. Skipping...")
                continue

            region_key = f"et_{iet}_eta_{ieta}"
            test_indices, _ = self.manifest.get_region_splits(region_key)

            data, target, _ = self.trainer.full_dataset[index]
            log.info(f"Using {data_path} for region iet{iet}.ieta{ieta}")

            if hasattr(self.aggregator, "get_best_model_for_region"):
                log.info("Getting best model for the region")
                best_model, best_rep, mean_sp, std_sp = self.aggregator.get_best_model_for_region(iet, ieta)
            else:
                raise Exception("Could not load the best model for et and eta region (iet={}, ieta={})".format(iet, ieta))
            self.model.load_state_dict(best_model["best_weights"])

            target_pd = self._get_target_pd(iet, ieta)
            result = self.evaluator.evaluate(
                model=self.model,
                data=data,
                target=target,
                test_indices=test_indices,
                ring_column_indices=self.trainer.full_dataset.ring_column_indices,
                device=self.trainer.device,
                target_pd=target_pd,
            )

            self.efficiency_plotter.plot_regional_efficiency(
                result=result,
                iet=iet,
                ieta=ieta,
                plot_dir=validation_dir,
                config_name=self.config_instance.config_name,
            )
            all_results.append(result)

        if all_results:
            self.efficiency_plotter.generate_global_plots(
                all_results,
                validation_dir,
                self.config_instance.config_name,
            )
