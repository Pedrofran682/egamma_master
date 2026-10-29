import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


class DistanceMetricsETL:
    """ETL processor for distance metric CSV tables generated during validation."""

    def __init__(
        self,
        folder_paths: Union[str, Path, List[Union[str, Path]]],
        file_pattern: str = "*distance_metrics_*.csv",
    ) -> None:
        """Initializes the ETL with one or multiple folder paths and file pattern.

        Args:
            folder_paths: Directory path or list of directory paths to scan.
            file_pattern: Filename glob pattern to match distance metric files.
        """
        if isinstance(folder_paths, (str, Path)):
            self._folder_paths = [Path(folder_paths)]
        else:
            self._folder_paths = [Path(p) for p in folder_paths]

        self._file_pattern = file_pattern
        self._discovered_files: List[Path] = self._discover_files()

    @property
    def folder_paths(self) -> List[Path]:
        return self._folder_paths

    @property
    def file_pattern(self) -> str:
        return self._file_pattern

    @property
    def discovered_files(self) -> List[Path]:
        return self._discovered_files

    def _discover_files(self) -> List[Path]:
        matched_files: List[Path] = []
        for folder in self._folder_paths:
            if not folder.exists():
                continue

            target_dirs = []
            tables_dir = folder / "Tables"
            if tables_dir.is_dir():
                target_dirs.append(tables_dir)
            elif folder.name == "Tables":
                target_dirs.append(folder)
            else:
                target_dirs.append(folder)
                sub_tables = list(folder.glob("**/Tables"))
                target_dirs.extend([d for d in sub_tables if d.is_dir()])

            for d in set(target_dirs):
                matched_files.extend(sorted(d.glob(self._file_pattern)))

        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        for f in matched_files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(f)

        return unique_files

    @staticmethod
    def extract_et_eta(file_path: Union[str, Path]) -> Tuple[int, int]:
        """Extracts integer (ET, ETA) indices from filename.

        Args:
            file_path: Path or filename containing 'et' and 'eta' identifiers.

        Returns:
            Tuple of (et, eta) integers.

        Raises:
            ValueError: If pattern cannot be found.
        """
        name = Path(file_path).name
        match = re.search(r"et_?(\d+).*?eta_?(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Could not parse ET and ETA from filename: {name}")

    @staticmethod
    def extract_execution_folder(file_path: Union[str, Path]) -> str:
        """Extracts the parent execution folder of the 'Tables' directory.

        Args:
            file_path: Path to the target CSV file.

        Returns:
            Name of the execution directory containing 'Tables', or immediate parent.
        """
        path = Path(file_path).resolve()
        if path.parent.name == "Tables":
            return path.parent.parent.name
        for parent in path.parents:
            if (parent / "Tables").is_dir():
                return parent.name
        return path.parent.name

    @staticmethod
    def extract_metric_type(file_path: Union[str, Path]) -> str:
        """Extracts the distance metric comparison descriptor from filename."""
        name = Path(file_path).stem
        match = re.search(r"distance_metrics_(.+)$", name)
        if match:
            return match.group(1)
        return name

    def load_single_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads a single CSV, parsing and appending ET, ETA, and ExecutionFolder columns.

        Args:
            file_path: Path to the CSV file.

        Returns:
            DataFrame with enriched columns.
        """
        path = Path(file_path)
        df = pd.read_csv(path)
        et, eta = self.extract_et_eta(path)
        execution_folder = self.extract_execution_folder(path)
        metric_type = self.extract_metric_type(path)

        df["ET"] = et
        df["ETA"] = eta
        df["ExecutionFolder"] = execution_folder
        df["MetricType"] = metric_type
        return df

    def extract_and_consolidate(self) -> pd.DataFrame:
        """Loads and consolidates all discovered CSV files into a unified DataFrame.

        Returns:
            Consolidated DataFrame with all records and added columns.
        """
        if not self._discovered_files:
            return pd.DataFrame(columns=["ET", "ETA", "ExecutionFolder", "MetricType"])

        dfs = [self.load_single_csv(f) for f in self._discovered_files]
        consolidated = pd.concat(dfs, ignore_index=True)
        return consolidated

    def get_summary(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generates a summary of loaded records grouped by ExecutionFolder, ET, and ETA."""
        target_df = df if df is not None else self.extract_and_consolidate()
        if target_df.empty:
            return pd.DataFrame()

        summary = (
            target_df.groupby(["ExecutionFolder", "ET", "ETA", "MetricType"])
            .size()
            .reset_index(name="RecordCount")
        )
        return summary

