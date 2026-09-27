import glob
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import awkward as ak
import numpy as np
import uproot

log = logging.getLogger(__name__)

DEFAULT_ETA_BINS: List[float] = [0.0, 0.8, 1.37, 1.54, 2.37, 2.5, 999.0]
DEFAULT_ET_BINS: List[float] = [15000.0, 20000.0, 30000.0, 40000.0, 50000.0, 999999000.0]

DEFAULT_TARGET_COLUMNS: List[str] = [
    "avgmu", "ph_et", "ph_eta", "ph_phi", "ph_Reta", "ph_Rphi",
    "ph_e237", "ph_e277", "ph_Rhad", "ph_Rhad1", "ph_weta1",
    "ph_weta2", "ph_f1", "ph_f3", "ph_f1core", "ph_fracs1",
    "ph_f3core", "ph_Eratio", "ph_deltaE", "ph_ethad", "ph_wtots1",
    "ph_ethad1", "ph_calo_eta", "ph_calo_phi", "ph_calo_et",
    "ph_calo_etaBE2", "ph_calo_e", "ph_hasCalo", "ph_ptcone20",
    "ph_ptcone30", "ph_ptcone40", "ph_ptvarcone20", "ph_ptvarcone30",
    "ph_ptvarcone40", "ph_isIsoFixedLoose", "ph_passCaloIso",
    "ph_passTrkIso", "ph_tight", "ph_medium", "ph_loose",
    "ph_nGoodVtx", "ph_nPileupPrimaryVtx", "trig_L1_ph_eta",
    "trig_L1_ph_phi", "trig_L1_ph_emClus", "trig_L1_ph_roi_et",
    "trig_L1_ph_emIso", "trig_L1_ph_hadCore", "trig_L1_ph_tauClus",
    "trig_L1eFex_ph_eta", "trig_L1eFex_ph_phi",
    "trig_L1eFex_ph_roi_et", "trig_L1eFex_ph_wstot",
    "trig_L1eFex_ph_reta", "trig_L1eFex_ph_rhad", "trig_L2_calo_et",
    "trig_L2_calo_eta", "trig_L2_calo_phi", "trig_L2_calo_e237",
    "trig_L2_calo_e277", "trig_L2_calo_fracs1", "trig_L2_calo_weta2",
    "trig_L2_calo_ehad1", "trig_L2_calo_emaxs1",
    "trig_L2_calo_e2tsts1", "trig_L2_calo_wstot",
    "trig_L2_calo_rings", "mc_hasMC",
    "mc_eta", "mc_phi", "mc_pt", "mc_isTop", "mc_isQuark",
    "mc_isParton", "mc_isMeson", "mc_isTau", "mc_isMuon",
    "mc_isPhoton", "mc_isElectron", "mc_origin", "mc_type", "mc_pdgID",
    "mc_status",
]


class RootDatasetGenerator:
    """Extracts, filters, and partitions calorimeter events from ROOT files into NPZ files."""

    def __init__(
        self,
        tree_name: str = "run_450000/HLT/EgammaMon/summary/events",
        vector_col: str = "trig_L2_calo_rings",
        vector_dim: int = 100,
        target_columns: List[str] | None = None,
        eta_bins: Sequence[float] | None = None,
        et_bins: Sequence[float] | None = None,
        num_splits: int = 2,
        uproot_workers: int = 4,
        compressed: bool = False,
    ) -> None:
        """Initializes RootDatasetGenerator.

        Args:
            tree_name: Name/path of TTree inside the ROOT files.
            vector_col: Column name containing vector calorimeter ring data.
            vector_dim: Expected dimensionality of padded vector data.
            target_columns: List of ROOT branch names to filter and load.
            eta_bins: Sequence of pseudorapidity bin boundaries.
            et_bins: Sequence of transverse energy bin boundaries in MeV.
            num_splits: Number of file batches to split the input files into.
            uproot_workers: Thread worker count for parallel ROOT decompression.
            compressed: Whether to save output NPZ archives using compression.
        """
        self.tree_name = tree_name
        self.vector_col = vector_col
        self.vector_dim = vector_dim
        self.target_columns = target_columns if target_columns is not None else DEFAULT_TARGET_COLUMNS
        self.eta_bins = list(eta_bins) if eta_bins is not None else DEFAULT_ETA_BINS
        self.et_bins = list(et_bins) if et_bins is not None else DEFAULT_ET_BINS
        self.num_splits = max(1, num_splits)
        self.uproot_workers = uproot_workers
        self.compressed = compressed

    def _convert_batch_to_numpy(
        self, events: ak.Array
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
        """Converts an Awkward record batch into contiguous 2D NumPy arrays in a single pass.

        Args:
            events: Awkward Array loaded from ROOT files.

        Returns:
            Tuple containing:
                - 2D dataset matrix (N, total_features)
                - List of feature column names
                - 1D array of absolute eta values
                - 1D array of ET values
                - 1D array of MC type values
        """
        features_list: List[str] = []
        columns_data: List[np.ndarray] = []

        rings_padded = ak.to_numpy(
            ak.fill_none(ak.pad_none(events[self.vector_col], self.vector_dim, axis=1), 0.0)[:, : self.vector_dim]
        )

        for field in events.fields:
            if field == self.vector_col:
                for idx in range(self.vector_dim):
                    features_list.append(f"{field}_{idx}")
                columns_data.append(rings_padded)
            else:
                features_list.append(field)
                arr = ak.to_numpy(events[field])
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                columns_data.append(arr)

        batch_data = np.hstack(columns_data)
        eta_arr = np.abs(ak.to_numpy(events["trig_L2_calo_eta"]))
        et_arr = ak.to_numpy(events["trig_L2_calo_et"])
        mc_type_arr = ak.to_numpy(events["mc_type"])

        return batch_data, features_list, eta_arr, et_arr, mc_type_arr

    def _filter_and_save_regions(
        self,
        batch_idx: int,
        batch_data: np.ndarray,
        feature_names: List[str],
        eta_arr: np.ndarray,
        et_arr: np.ndarray,
        mc_type_arr: np.ndarray,
        file_prefix: str,
        is_perf_jf: bool,
        output_dir: Path,
    ) -> List[Path]:
        """Vectorizes regional bin filtering and writes partitioned NPZ files.

        Args:
            batch_idx: Current file batch index.
            batch_data: Pre-stacked 2D NumPy matrix of event features.
            feature_names: List of column feature names.
            eta_arr: 1D array of absolute eta coordinates.
            et_arr: 1D array of transverse energy in MeV.
            mc_type_arr: 1D array of particle generator MC types.
            file_prefix: Dataset identifier prefix used for naming.
            is_perf_jf: Whether dataset belongs to background jet-fake sample.
            output_dir: Output directory path.

        Returns:
            List of generated NPZ file paths.
        """
        saved_paths: List[Path] = []
        features_np = np.array(feature_names, dtype=str)

        for eta_idx in range(len(self.eta_bins) - 1):
            eta_min, eta_max = self.eta_bins[eta_idx], self.eta_bins[eta_idx + 1]

            for et_idx in range(len(self.et_bins) - 1):
                et_min, et_max = self.et_bins[et_idx], self.et_bins[et_idx + 1]

                kinematic_mask = (
                    (eta_arr >= eta_min) & (eta_arr < eta_max) &
                    (et_arr >= et_min) & (et_arr < et_max)
                )

                if not np.any(kinematic_mask):
                    continue

                if is_perf_jf:
                    region_mask = kinematic_mask & ~np.isin(mc_type_arr, [13, 14, 15])
                    target_val = 0
                else:
                    region_mask = kinematic_mask & np.isin(mc_type_arr, [13, 14, 15])
                    target_val = 1

                event_count = np.count_nonzero(region_mask)
                if event_count == 0:
                    continue

                region_data = batch_data[region_mask]
                target_array = np.full(event_count, target_val, dtype=np.int32)
                source_array = np.full(event_count, file_prefix, dtype=f"<U{len(file_prefix)}")

                out_filename = f"{file_prefix}_part{batch_idx}.et{et_idx}.eta{eta_idx}.npz"
                out_path = output_dir / out_filename

                save_kwargs: Dict[str, np.ndarray] = {
                    "feature": features_np,
                    "data": region_data,
                    "target": target_array,
                    "source": source_array,
                }

                if self.compressed:
                    np.savez_compressed(out_path, **save_kwargs)
                else:
                    np.savez(out_path, **save_kwargs)

                log.info("Saved: %s (%d events)", out_filename, event_count)
                saved_paths.append(out_path)

        return saved_paths

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        file_prefix: str,
    ) -> List[Path]:
        """Processes all ROOT files in an input directory into binned NPZ partitions.

        Args:
            input_dir: Directory containing .root files.
            output_dir: Root output directory.
            file_prefix: Prefix identifier for the dataset.

        Returns:
            List of generated NPZ file paths.
        """
        input_path = Path(input_dir)
        dest_dir = Path(output_dir) / file_prefix
        dest_dir.mkdir(parents=True, exist_ok=True)

        all_files = sorted(glob.glob(f"{input_path}/*.root"))
        if not all_files:
            log.warning("No .root files found in %s", input_path)
            return []

        log.info("Found %d ROOT files in %s", len(all_files), input_path)
        is_perf_jf = "perf_JF" in file_prefix

        file_batches = np.array_split(all_files, self.num_splits)
        generated_files: List[Path] = []

        for batch_idx, batch in enumerate(file_batches):
            if len(batch) == 0:
                continue

            file_paths = [f"{f}:{self.tree_name}" for f in batch]
            log.info(
                "Processing batch %d/%d with %d files...",
                batch_idx + 1,
                self.num_splits,
                len(file_paths),
            )

            events = uproot.concatenate(
                file_paths,
                filter_name=self.target_columns,
                library="ak",
                num_workers=self.uproot_workers,
            )

            (
                batch_data,
                features_list,
                eta_arr,
                et_arr,
                mc_type_arr,
            ) = self._convert_batch_to_numpy(events)

            paths = self._filter_and_save_regions(
                batch_idx=batch_idx,
                batch_data=batch_data,
                feature_names=features_list,
                eta_arr=eta_arr,
                et_arr=et_arr,
                mc_type_arr=mc_type_arr,
                file_prefix=file_prefix,
                is_perf_jf=is_perf_jf,
                output_dir=dest_dir,
            )
            generated_files.extend(paths)

        return generated_files
