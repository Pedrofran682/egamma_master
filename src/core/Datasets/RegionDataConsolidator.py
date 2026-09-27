import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Pattern, Tuple
import numpy as np

log = logging.getLogger(__name__)


def _consolidate_single_group(
    et: str,
    eta: str,
    paths: List[Path],
    output_dir: Path,
    compressed: bool,
) -> Path | None:
    """Merges NPZ partitions for a single (et, eta) group and saves consolidated NPZ.

    Args:
        et: Transverse energy bin index string.
        eta: Pseudorapidity bin index string.
        paths: List of NPZ file paths belonging to this group.
        output_dir: Directory where the consolidated file will be written.
        compressed: Whether to compress the output archive.

    Returns:
        Path to the saved consolidated NPZ file, or None if no valid data was found.
    """
    data_list: List[np.ndarray] = []
    source_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    feature_array: np.ndarray | None = None

    for path in paths:
        try:
            with np.load(path, allow_pickle=True) as archive:
                if "data" not in archive.files:
                    continue

                current_data = archive["data"]
                data_list.append(current_data)

                if "source" in archive.files:
                    source_list.append(archive["source"])
                else:
                    source_list.append(np.full(current_data.shape[0], path.name))

                if "target" in archive.files:
                    target_list.append(archive["target"])

                if feature_array is None and "feature" in archive.files:
                    feature_array = archive["feature"]
        except Exception as exc:
            log.warning("Failed to load %s: %s", path.name, exc)
            continue

    if not data_list:
        log.warning("No valid data found for group et=%s, eta=%s", et, eta)
        return None

    save_dict: Dict[str, np.ndarray] = {
        "data": np.concatenate(data_list, axis=0),
        "source": np.concatenate(source_list, axis=0).astype(str),
    }

    if target_list and len(target_list) == len(data_list):
        save_dict["target"] = np.concatenate(target_list, axis=0)

    if feature_array is not None:
        save_dict["feature"] = feature_array

    output_filename = f"consolidated.et{et}.eta{eta}.npz"
    output_path = output_dir / output_filename

    if compressed:
        np.savez_compressed(output_path, **save_dict)
    else:
        np.savez(output_path, **save_dict)

    log.info("Saved consolidated group: %s (%d events)", output_filename, save_dict["data"].shape[0])
    return output_path


class RegionDataConsolidator:
    """Consolidates partitioned calorimeter NPZ data files by kinematic (ET, eta) bins."""

    def __init__(
        self,
        base_dir: str | Path,
        output_dir: str | Path,
        name_pattern: str = r"\.et(\d+)\.eta(\d+).*\.npz$",
        ignore_pattern: str = r"\.sys\.v\d+",
        max_workers: int | None = 1,
        compressed: bool = True,
    ) -> None:
        """Initializes RegionDataConsolidator.

        Args:
            base_dir: Source directory containing partitioned NPZ files.
            output_dir: Destination directory where consolidated files are stored.
            name_pattern: Regex pattern capturing et and eta indices from filenames.
            ignore_pattern: Regex pattern identifying files to skip.
            max_workers: Number of parallel worker processes. Set to None for all CPUs.
            compressed: Whether to write compressed archives via np.savez_compressed.
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.name_regex: Pattern[str] = re.compile(name_pattern)
        self.ignore_regex: Pattern[str] = re.compile(ignore_pattern)
        self.max_workers = max_workers
        self.compressed = compressed

    def discover_groups(self) -> Dict[Tuple[str, str], List[Path]]:
        """Scans base directory and groups valid NPZ files by (ET, eta) indices.

        Returns:
            Dictionary mapping (et_idx, eta_idx) to matching file paths.
        """
        groups: Dict[Tuple[str, str], List[Path]] = {}

        for file_path in self.base_dir.rglob("*.npz"):
            if self.ignore_regex.search(str(file_path)):
                continue

            match = self.name_regex.search(file_path.name)
            if not match:
                continue

            key = (match.group(1), match.group(2))
            groups.setdefault(key, []).append(file_path)

        for key in groups:
            groups[key].sort()

        return dict(sorted(groups.items()))

    def consolidate_group(self, et: str, eta: str, paths: List[Path]) -> Path | None:
        """Consolidates a single (et, eta) group synchronously.

        Args:
            et: Transverse energy bin index string.
            eta: Pseudorapidity bin index string.
            paths: List of NPZ paths belonging to this kinematic bin.

        Returns:
            Path to the saved consolidated NPZ file, or None if skipped.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return _consolidate_single_group(
            et=et,
            eta=eta,
            paths=paths,
            output_dir=self.output_dir,
            compressed=self.compressed,
        )

    def run(self) -> List[Path]:
        """Executes consolidation across all discovered kinematic groups.

        Returns:
            List of generated consolidated file paths.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        groups = self.discover_groups()

        if not groups:
            log.warning("No matching groups found in %s", self.base_dir)
            return []

        log.info("Discovered %d kinematic groups to consolidate", len(groups))

        if self.max_workers == 1:
            results = [
                self.consolidate_group(et, eta, paths)
                for (et, eta), paths in groups.items()
            ]
            return [p for p in results if p is not None]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    _consolidate_single_group,
                    et,
                    eta,
                    paths,
                    self.output_dir,
                    self.compressed,
                )
                for (et, eta), paths in groups.items()
            ]
            results = [f.result() for f in futures]

        return [p for p in results if p is not None]
