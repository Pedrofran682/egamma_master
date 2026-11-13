import logging
import os
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.core.Interfaces.BaseHyperparameterTuner import BaseHyperparameterTuner
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Tuning.TuningConfiguration import TuningConfiguration
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration

log = logging.getLogger(__name__)


class GridSearchTuner(BaseHyperparameterTuner):
    """Grid search tuner for optimizing model hyperparameters across folds and regions."""

    def __init__(self, tuning_config: TuningConfiguration) -> None:
        """Initializes the grid search tuner.

        Args:
            tuning_config: Validated tuning configuration.
        """
        self.tuning_config: TuningConfiguration = tuning_config
        self._trials: List[Tuple[str, NeuralRingerTrainerConfiguration, Dict[str, Any]]] = (
            self.tuning_config.build_trial_configurations()
        )

    def generate_trials(self) -> List[Tuple[str, NeuralRingerTrainerConfiguration]]:
        """Generates list of trial tags and associated trainer configurations.

        Returns:
            List of tuples containing unique trial identifier and trainer config.
        """
        return [(tag, cfg) for tag, cfg, _ in self._trials]

    def run(self) -> pd.DataFrame:
        """Executes hyperparameter trials and aggregates performance metrics.

        Returns:
            DataFrame containing evaluated trials sorted by mean validation SP index.
        """
        records: List[Dict[str, Any]] = []
        total_trials = len(self._trials)
        log.info(f"Starting grid search hyperparameter tuning with {total_trials} trials.")

        output_dir = (
            self.tuning_config.output_dir
            or f"results/tuning_{os.path.basename(self.tuning_config.base_config).replace('.yaml', '')}"
        )
        os.makedirs(output_dir, exist_ok=True)

        for index, (trial_tag, trial_cfg, metadata) in enumerate(self._trials, start=1):
            log.info(f"--- [Trial {index}/{total_trials}] {trial_tag} ---")
            trial_path = str(trial_cfg.results_folder_path)

            try:
                aggregator = ResultAggregator(trial_path)
                already_completed = self._is_trial_complete(aggregator, trial_cfg)

                if not already_completed:
                    trainer = NeuralRingerTrainer(trial_cfg)
                    trainer.run()
                    aggregator = ResultAggregator(trial_path)

                metrics = self._evaluate_trial_metrics(aggregator, trial_cfg)
                record = {**metadata, **metrics}
                records.append(record)
            except Exception as e:
                log.error(f"Error executing trial {trial_tag}: {e}")
                record = {
                    **metadata,
                    "mean_sp": 0.0,
                    "std_sp": 0.0,
                    "best_sp": 0.0,
                    "best_pd": 0.0,
                    "best_fa": 1.0,
                    "status": f"FAILED: {e}",
                }
                records.append(record)

        results_df = pd.DataFrame(records)
        if not results_df.empty and "mean_sp" in results_df.columns:
            results_df = results_df.sort_values(by=["mean_sp", "std_sp"], ascending=[False, True]).reset_index(
                drop=True
            )

        log.info("Hyperparameter tuning execution completed across all trial sets.")
        return results_df

    def _is_trial_complete(
        self, aggregator: ResultAggregator, config: NeuralRingerTrainerConfiguration
    ) -> bool:
        """Verifies if all configured regions have recorded results.

        Args:
            aggregator: ResultAggregator for trial output directory.
            config: Trainer configuration of the trial.

        Returns:
            True if all regions have result files, False otherwise.
        """
        if not os.path.isdir(aggregator.directory_path):
            return False

        for et in config.et_range_idx:
            for eta in config.eta_range_idx:
                if not aggregator.has_region(et, eta):
                    return False
        return True

    def _evaluate_trial_metrics(
        self, aggregator: ResultAggregator, config: NeuralRingerTrainerConfiguration
    ) -> Dict[str, Any]:
        """Extracts validation performance metrics across configured regions.

        Args:
            aggregator: ResultAggregator instance for trial outputs.
            config: Trial configuration instance.

        Returns:
            Dictionary containing aggregated performance statistics.
        """
        region_means: List[float] = []
        region_stds: List[float] = []
        region_best_sps: List[float] = []
        region_best_pds: List[float] = []
        region_best_fas: List[float] = []

        for et in config.et_range_idx:
            for eta in config.eta_range_idx:
                if aggregator.has_region(et, eta):
                    best_model, _, mean_sp, std_sp = aggregator.get_best_model_for_region(et, eta)
                    region_means.append(mean_sp)
                    region_stds.append(std_sp)
                    region_best_sps.append(float(best_model.get("best_sp_value", 0.0)))
                    region_best_pds.append(float(best_model.get("best_pd_value", 0.0)))
                    region_best_fas.append(float(best_model.get("best_fa_value", 0.0)))

        return {
            "mean_sp": float(pd.Series(region_means).mean()) if region_means else 0.0,
            "std_sp": float(pd.Series(region_stds).mean()) if region_stds else 0.0,
            "best_sp": float(pd.Series(region_best_sps).max()) if region_best_sps else 0.0,
            "best_pd": float(pd.Series(region_best_pds).mean()) if region_best_pds else 0.0,
            "best_fa": float(pd.Series(region_best_fas).mean()) if region_best_fas else 0.0,
            "status": "SUCCESS",
        }
