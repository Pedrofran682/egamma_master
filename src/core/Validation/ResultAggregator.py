import logging
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.core.Interfaces.BaseResultAggregator import BaseResultAggregator

log = logging.getLogger()


class ResultAggregator(BaseResultAggregator):
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

    def _discover_region_files(self) -> Dict[str, List[pathlib.Path]]:
        files = [p for p in self.directory_path.iterdir() if p.is_file() and p.suffix == ".pkl"]
        groups: Dict[str, List[pathlib.Path]] = defaultdict(list)
        for file_path in files:
            match = re.search(r"(iet\d+\.ieta\d+)", file_path.name)
            group_key = match.group(1) if match else "undefined_region"
            groups[group_key].append(file_path)
        return groups

    @staticmethod
    def _load_files_to_df(file_list: List[pathlib.Path]) -> pd.DataFrame:
        df_list = []
        for f in file_list:
            try:
                loaded = pd.read_pickle(f)
                if isinstance(loaded, dict):
                    loaded = [loaded]
                elif isinstance(loaded, list):
                    loaded = [
                        {
                            **result,
                            **(
                                {"repeat": result.get("repeat", result.get("reapet"))}
                                if isinstance(result, dict)
                                and ("repeat" in result or "reapet" in result)
                                else {}
                            ),
                        }
                        for result in loaded
                    ]
                df_item = pd.DataFrame(loaded)
                if not df_item.empty:
                    if "repeat" not in df_item.columns and "reapet" in df_item.columns:
                        df_item["repeat"] = df_item["reapet"]
                    elif "repeat" in df_item.columns and "reapet" in df_item.columns:
                        df_item["repeat"] = df_item["repeat"].combine_first(df_item["reapet"])
                    df_list.append(df_item)
            except Exception as e:
                log.warning(f"Skipping unreadable result file {f}: {e}")
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    def group_and_concat(self) -> Dict[str, pd.DataFrame]:
        """Scans directory for .pkl files, groups them by region, and concatenates statistics.

        Returns:
            Dictionary mapping region keys (e.g. 'iet1.ieta1') to consolidated DataFrames.
        """
        groups = self._discover_region_files()
        grouped_dfs: Dict[str, pd.DataFrame] = {
            group_key: self._load_files_to_df(file_list)
            for group_key, file_list in groups.items()
        }
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
        if self._grouped_dfs is not None:
            return region_key in self._grouped_dfs and not self._grouped_dfs[region_key].empty
        return bool(self._discover_region_files().get(region_key))

    def _get_region_dataframe(self, et: int, eta: int, cache: bool = True) -> pd.DataFrame:
        region_key = f"iet{et}.ieta{eta}"
        if not cache and self._grouped_dfs is None:
            region_files = self._discover_region_files().get(region_key, [])
            if not region_files:
                raise KeyError(
                    f"No result records found for region '{region_key}' in {self.directory_path}."
                )
            region_df = self._load_files_to_df(region_files)
            if region_df.empty:
                raise KeyError(
                    f"No result records found for region '{region_key}' in {self.directory_path}."
                )
            return region_df

        if self._grouped_dfs is None:
            self._grouped_dfs = self.group_and_concat()
        if region_key not in self._grouped_dfs or self._grouped_dfs[region_key].empty:
            raise KeyError(
                f"No result records found for region '{region_key}' in {self.directory_path}."
            )
        return self._grouped_dfs[region_key]

    def get_best_model_for_region(
        self, et: int, eta: int, cache: bool = True
    ) -> Tuple[Dict[str, Any], int, float, float]:
        """Extracts the best model details specifically for a given kinematic region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.
            cache: Whether to load and cache all regions in memory (`True`) or load only
                the requested region on demand without retaining its DataFrame (`False`).

        Returns:
            A tuple containing:
                - Dictionary representation of the best model row (weights, metrics).
                - Best repetition index.
                - Mean SP value across folds for that repetition.
                - Standard deviation of the SP values across folds.

        Raises:
            KeyError: If no result records exist for the specified region.
        """
        return self.get_best_model_details(self._get_region_dataframe(et, eta, cache=cache))

    def get_best_training_metrics_for_region(
        self, et: int, eta: int, cache: bool = False
    ) -> Dict[str, Any]:
        """Extracts best-SP fold metrics and across-fold uncertainties for a kinematic region.

        Args:
            et: Transverse energy bin index.
            eta: Pseudorapidity bin index.
            cache: Whether to cache all regions in memory (`True`) or stream only this region (`False`).

        Returns:
            Dictionary with `pd`, `pd_uncertainty`, `sp`, `sp_uncertainty`, `pf`, `pf_uncertainty`,
            `repeat`, `fold`, `mean_rep_sp`, and `std_rep_sp`.
        """
        region_df = self._get_region_dataframe(et, eta, cache=cache)
        best_model, best_rep, mean_sp, std_sp = self.get_best_model_details(region_df)
        repeat_series = self._resolve_repeat_series(region_df)
        df_best_rep = region_df[repeat_series == best_rep]

        def _fold_std(col_name: str) -> float:
            if col_name not in df_best_rep.columns:
                return float("nan")
            valid = df_best_rep[col_name].dropna().astype(float)
            if len(valid) > 1:
                return float(valid.std(ddof=1))
            return 0.0 if len(valid) == 1 else float("nan")

        return {
            "repeat": best_rep,
            "fold": int(best_model.get("fold", -1)),
            "pd": float(best_model.get("best_pd_value", float("nan"))),
            "pd_uncertainty": _fold_std("best_pd_value"),
            "sp": float(best_model.get("best_sp_value", float("nan"))),
            "sp_uncertainty": _fold_std("best_sp_value"),
            "pf": float(best_model.get("best_fa_value", float("nan"))),
            "pf_uncertainty": _fold_std("best_fa_value"),
            "mean_rep_sp": mean_sp,
            "std_rep_sp": std_sp,
        }

    @staticmethod
    def _resolve_repeat_series(df: pd.DataFrame) -> pd.Series:
        if "repeat" in df.columns and "reapet" in df.columns:
            return df["repeat"].combine_first(df["reapet"]).rename("repeat")
        if "repeat" in df.columns:
            return df["repeat"].rename("repeat")
        if "reapet" in df.columns:
            return df["reapet"].rename("repeat")
        raise KeyError("Results DataFrame must contain 'repeat' or 'reapet' column.")

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
        repeat_series = ResultAggregator._resolve_repeat_series(df)
        df_stats = (
            df.groupby(repeat_series)["best_sp_value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        best_init_idx = df_stats["mean"].idxmax()
        best_repetition = int(df_stats.loc[best_init_idx, "repeat"])
        best_mean_sp = float(df_stats.loc[best_init_idx, "mean"])
        best_std_sp = float(df_stats.loc[best_init_idx, "std"])

        df_best_rep = df[repeat_series == best_repetition]
        best_model_idx = df_best_rep["best_sp_value"].idxmax()
        best_model = df.loc[best_model_idx].to_dict()
        if "repeat" not in best_model or pd.isna(best_model.get("repeat")):
            if "reapet" in best_model:
                best_model["repeat"] = best_model["reapet"]

        return best_model, best_repetition, best_mean_sp, best_std_sp
