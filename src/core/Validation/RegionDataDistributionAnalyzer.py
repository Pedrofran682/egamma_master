import logging
import os
import pathlib
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from src.core.Interfaces.BaseEgammaDataset import BaseEgammaDataset
from src.core.Datasets.EgammaNpzDataset import EgammaNpzDataset
from src.core.Plotting.RegionDistributionPlotter import RegionDistributionPlotter
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_et_eta, get_instance

log = logging.getLogger(__name__)


class RegionDataDistributionAnalyzer:
    """Extracts, summarizes, and visualizes class distribution across ET and eta regions.

    Attributes:
        config_path: Path to YAML training configuration.
        config: Parsed NeuralRingerTrainerConfiguration.
        data_path: Optional explicit data directory override.
        output_dir: Directory path for exported plots and CSVs.
        dataset: Instantiated EgammaNpzDataset.
        plotter: RegionDistributionPlotter instance.
    """

    def __init__(
        self,
        config_path: str,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """Initializes RegionDataDistributionAnalyzer.

        Args:
            config_path: Path to configuration YAML file.
            data_path: Optional override for the dataset NPZ directory path.
            output_dir: Destination folder path for generated plots and data.
        """
        self.config_path: pathlib.Path = pathlib.Path(config_path)
        self.config: NeuralRingerTrainerConfiguration = self._load_config(config_path)
        self.data_path: Optional[pathlib.Path] = pathlib.Path(data_path) if data_path else None
        self.output_dir: pathlib.Path = self._resolve_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset: BaseEgammaDataset = self._initialize_dataset()
        self.plotter: RegionDistributionPlotter = RegionDistributionPlotter()

    @staticmethod
    def _load_config(config_path: str) -> NeuralRingerTrainerConfiguration:
        """Loads and parses the YAML configuration.

        Args:
            config_path: Path to YAML file.

        Returns:
            Validated NeuralRingerTrainerConfiguration instance.
        """
        with open(config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            return NeuralRingerTrainerConfiguration.model_validate(yaml_data)

    def _resolve_output_dir(self, output_dir: Optional[str]) -> pathlib.Path:
        """Determines the target output directory path.

        Args:
            output_dir: User-specified output path or None.

        Returns:
            Resolved Path object for output figures.
        """
        if output_dir is not None:
            return pathlib.Path(output_dir)
        config_name = self.config.config_name or self.config_path.stem
        return pathlib.Path("Plots") / config_name / "DataDistribution"

    def _initialize_dataset(self) -> EgammaNpzDataset:
        """Instantiates the dataset configured in YAML, applying path overrides if provided.

        Returns:
            Configured dataset instance.
        """
        if self.data_path is not None:
            self.config.dataset.parameters["drive_path"] = str(self.data_path)

        dataset: EgammaNpzDataset = get_instance(self.config.dataset)
        dataset.config = self.config
        return dataset

    def collect_distribution(self) -> pd.DataFrame:
        """Scans dataset files and computes class 0 and class 1 counts per region.

        Returns:
            DataFrame containing aggregated counts and ratios per ET and eta region.
        """
        region_records: List[Dict[str, object]] = []
        target_file_paths = self.dataset.file_paths

        log.info(f"Scanning {len(target_file_paths)} files for regional class distribution...")

        for file_path in target_file_paths:
            try:
                transverse_energy_bin, pseudorapidity_bin = get_et_eta(file_path)
                with np.load(file_path, allow_pickle=True) as sample_archive:
                    _, sample_labels = self.dataset.filter_events(sample_archive)

                background_count = int(np.sum(sample_labels == 0))
                signal_count = int(np.sum(sample_labels == 1))
                total_events = int(len(sample_labels))
                signal_fraction = float(signal_count / total_events) if total_events > 0 else 0.0

                et_interval = self.plotter.et_intervals.get(int(transverse_energy_bin), f"ET {transverse_energy_bin}")
                eta_interval = self.plotter.eta_intervals.get(int(pseudorapidity_bin), f"eta {pseudorapidity_bin}")

                region_records.append(
                    {
                        "et": int(transverse_energy_bin),
                        "eta": int(pseudorapidity_bin),
                        "et_range": et_interval,
                        "eta_range": eta_interval,
                        "background_count": background_count,
                        "signal_count": signal_count,
                        "total_events": total_events,
                        "signal_fraction": signal_fraction,
                        "file_path": file_path,
                    }
                )
            except Exception as e:
                log.warning(f"Failed to process file {file_path}: {e}")

        distribution_dataframe = pd.DataFrame(region_records)
        if not distribution_dataframe.empty:
            distribution_dataframe = distribution_dataframe.sort_values(by=["et", "eta"]).reset_index(drop=True)
        return distribution_dataframe

    def run(self, file_format: str = "pdf") -> pd.DataFrame:
        """Executes full extraction, exports CSV summary, and renders 2D heatmaps.

        Args:
            file_format: Image file format for plots ('pdf', 'png').

        Returns:
            DataFrame containing summarized region distributions.
        """
        region_distribution_dataframe = self.collect_distribution()
        if region_distribution_dataframe.empty:
            log.warning("No data found to plot.")
            return region_distribution_dataframe

        csv_summary_path = self.output_dir / "region_data_distribution.csv"
        region_distribution_dataframe.to_csv(csv_summary_path, index=False)
        log.info(f"Saved distribution summary to {csv_summary_path}")

        generated_plots = self.plotter.plot(
            region_distribution_dataframe=region_distribution_dataframe,
            output_dir=self.output_dir,
            file_format=file_format,
            subfolder="",
        )
        for plot_name, plot_path in generated_plots.items():
            log.info(f"Generated plot '{plot_name}': {plot_path}")

        return region_distribution_dataframe
