import logging
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import pandas as pd

log = logging.getLogger()


class ResultAggregator:
    """Discovers, groups, and aggregates model run results across kinematic regions.

    Attributes:
        directory_path: Directory path where results pickle files are located.
    """

    def __init__(self, directory_path: pathlib.Path | str) -> None:
        """Initializes ResultAggregator with a directory path.

        Args:
            directory_path: Folder containing fold/repeat result pickle archives.
        """
        self.directory_path: pathlib.Path = pathlib.Path(directory_path)
        self._grouped_dfs: Dict[str, pd.DataFrame] | None = None

    def group_and_concat(self) -> Dict[str, pd.DataFrame]:
        """Scans directory for .pkl files, groups them by region, and concatenates statistics.

        Returns:
            Dictionary mapping region keys (e.g. 'iet1.ieta1') to consolidated DataFrames.
        """
        files = [p for p in self.directory_path.iterdir() if p.is_file() and p.suffix == ".pkl"]
        groups: Dict[str, List[pathlib.Path]] = defaultdict(list)

        for file_path in files:
            match = re.search(r"(iet\d+\.ieta\d+)", file_path.name)
            group_key = match.group(1) if match else "undefined_region"
            groups[group_key].append(file_path)

        grouped_dfs: Dict[str, pd.DataFrame] = {}
        for group_key, file_list in groups.items():
            df_list = []
            for f in file_list:
                try:
                    loaded = pd.read_pickle(f)
                    df_item = pd.DataFrame(loaded)
                    if not df_item.empty:
                        df_list.append(df_item)
                except Exception as e:
                    log.warning(f"Skipping unreadable result file {f}: {e}")
            if df_list:
                grouped_dfs[group_key] = pd.concat(df_list, ignore_index=True)
            else:
                grouped_dfs[group_key] = pd.DataFrame()

        self._grouped_dfs = grouped_dfs
        return grouped_dfs

    def has_region(self, et: int, eta: int) -> bool:
        """Checks if result records exist for a specified kinematic region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.

        Returns:
            True if records exist for the region, False otherwise.
        """
        region_key = f"iet{et}.ieta{eta}"
        if self._grouped_dfs is None:
            self._grouped_dfs = self.group_and_concat()
        return region_key in self._grouped_dfs and not self._grouped_dfs[region_key].empty

    def get_best_model_for_region(
        self, et: int, eta: int
    ) -> Tuple[Dict[str, Any], int, float, float]:
        """Extracts the best model details specifically for a given kinematic region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.

        Returns:
            A tuple containing:
                - Dictionary representation of the best model row (weights, metrics).
                - Best repetition index.
                - Mean SP value across folds for that repetition.
                - Standard deviation of the SP values across folds.

        Raises:
            KeyError: If no result records exist for the specified region.
        """
        region_key = f"iet{et}.ieta{eta}"
        if self._grouped_dfs is None:
            self._grouped_dfs = self.group_and_concat()
        if region_key not in self._grouped_dfs or self._grouped_dfs[region_key].empty:
            raise KeyError(
                f"No result records found for region '{region_key}' in {self.directory_path}."
            )
        return self.get_best_model_details(self._grouped_dfs[region_key])

    @staticmethod
    def get_best_model_details(df: pd.DataFrame) -> Tuple[Dict[str, Any], int, float, float]:
        """Determines the top-performing initialization repetition and extracts its best fold model.

        Args:
            df: Consolidated DataFrame containing training result records.

        Returns:
            A tuple containing:
                - Dictionary representation of the best model row (weights, metrics).
                - Best repetition / initialization index.
                - Mean SP value across folds for that repetition.
                - Standard deviation of the SP values across folds.
        """
        df_stats = df.groupby("reapet")["best_sp_value"].agg(["mean", "std"]).reset_index()
        best_init_idx = df_stats["mean"].idxmax()
        best_repetition = int(df_stats.loc[best_init_idx, "reapet"])
        best_mean_sp = float(df_stats.loc[best_init_idx, "mean"])
        best_std_sp = float(df_stats.loc[best_init_idx, "std"])

        df_best_rep = df[df["reapet"] == best_repetition]
        best_model_idx = df_best_rep["best_sp_value"].idxmax()
        best_model = df.loc[best_model_idx].to_dict()

        return best_model, best_repetition, best_mean_sp, best_std_sp
