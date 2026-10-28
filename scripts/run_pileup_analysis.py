import argparse
import pathlib
import sys
import yaml
import numpy as np
import torch

from src.core.Datasets.SplitManifest import SplitManifest
from src.core.Plotting.PileupPlotter import PileupPlotter
from src.core.Trainers.NeuralRingerTrainer import NeuralRingerTrainer
from src.core.Validation.FastPhotonCutEvaluator import FastPhotonCutEvaluator
from src.core.Validation.HoldoutEvaluator import HoldoutEvaluator
from src.core.Validation.PileupAnalyzer import PileupAnalyzer
from src.core.Validation.ResultAggregator import ResultAggregator
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration
from src.utils import get_et_eta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ATLAS pileup sensitivity and detector drift analysis across <mu> and NPV."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to NeuralRinger YAML configuration.",
    )
    parser.add_argument(
        "--results_path",
        type=pathlib.Path,
        required=True,
        help="Path to trained model results directory containing fold outputs and split_manifest.json.",
    )
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        default=None,
        help="Optional override path for consolidated dataset directory.",
    )
    parser.add_argument(
        "--target_et",
        type=int,
        default=1,
        help="Target ET region bin index (default: 1).",
    )
    parser.add_argument(
        "--target_eta",
        type=int,
        default=1,
        help="Target eta region bin index (default: 1).",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("Plots/pileup_analysis"),
        help="Directory to export generated figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Graphic output format (default: png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    if args.data_path is not None and args.data_path.exists():
        config_dict["dataset"]["parameters"]["drive_path"] = str(args.data_path)

    config = NeuralRingerTrainerConfiguration.model_validate(config_dict)
    trainer = NeuralRingerTrainer(config)
    full_dataset = trainer.full_dataset

    region_file_idx = None
    for idx, path_str in enumerate(full_dataset.file_paths):
        iet, ieta = get_et_eta(path_str)
        if iet == args.target_et and ieta == args.target_eta:
            region_file_idx = idx
            break

    if region_file_idx is None:
        raise FileNotFoundError(
            f"No dataset file matching et={args.target_et}, eta={args.target_eta} found."
        )

    raw_features, raw_labels, region_file_path = full_dataset[region_file_idx]
    feature_names = full_dataset.feature_names
    region_label = f"iet{args.target_et}.ieta{args.target_eta}"

    manifest_path = args.results_path / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")

    manifest = SplitManifest(str(manifest_path))
    region_key = f"et_{args.target_et}_eta_{args.target_eta}"
    if region_key not in manifest.regions:
        raise KeyError(f"Region '{region_key}' not partitioned in {manifest_path}")

    test_indices = manifest.get_test_indices(region_key)
    test_features = raw_features[test_indices]
    test_labels = raw_labels[test_indices]

    print(f"Loaded region {region_label} from {region_file_path}")
    print(f"Test events: {len(test_labels):,} (Signal: {np.sum(test_labels == 1):,}, Background: {np.sum(test_labels == 0):,})")

    aggregator = ResultAggregator()
    best_model_info = aggregator.get_best_model(str(args.results_path), args.target_et, args.target_eta)
    if best_model_info is None:
        raise RuntimeError(f"Could not load best model checkpoint from {args.results_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = HoldoutEvaluator(device=device)
    raw_scores = evaluator.evaluate(best_model_info["model"], test_features)

    threshold = float(best_model_info.get("threshold", 0.0))
    model_preds = (raw_scores >= threshold).astype(int)

    athena_evaluator = FastPhotonCutEvaluator()
    athena_preds = athena_evaluator.evaluate(
        test_features,
        feature_names,
        working_point="tight",
        iet=args.target_et,
        ieta=args.target_eta,
    )

    plotter = PileupPlotter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mu_values = PileupAnalyzer.extract_feature(test_features, feature_names, "avgmu")
    except KeyError:
        print("Feature 'avgmu' not found in dataset. Skipping pileup analysis.")
        return

    sig_eff_model = PileupAnalyzer.compute_efficiency_vs_mu(
        mu_values, test_labels, model_preds, bins=10, is_signal=True
    )
    sig_eff_athena = PileupAnalyzer.compute_efficiency_vs_mu(
        mu_values, test_labels, athena_preds, bins=10, is_signal=True
    )

    models_data_sig = [
        {
            "label": "Athena Tight Cut",
            "bin_centers": sig_eff_athena["bin_centers"],
            "efficiencies": sig_eff_athena["efficiencies"],
            "uncertainties": sig_eff_athena["uncertainties"],
            "color": "black",
            "marker": "s",
            "is_ml": False,
            "is_reference": True,
        },
        {
            "label": f"Neural Model (Fold {best_model_info.get('fold_idx', 0)})",
            "bin_centers": sig_eff_model["bin_centers"],
            "efficiencies": sig_eff_model["efficiencies"],
            "uncertainties": sig_eff_model["uncertainties"],
            "color": "tab:blue",
            "marker": "o",
            "is_ml": True,
            "is_reference": False,
        },
    ]

    plotter.plot_efficiency_vs_mu(
        models_data=models_data_sig,
        region_label=region_label,
        is_signal=True,
        output_path=args.output_dir / f"efficiency_vs_mu_signal.{args.format}",
        file_format=args.format,
    )

    bg_eff_model = PileupAnalyzer.compute_efficiency_vs_mu(
        mu_values, test_labels, model_preds, bins=10, is_signal=False
    )
    bg_eff_athena = PileupAnalyzer.compute_efficiency_vs_mu(
        mu_values, test_labels, athena_preds, bins=10, is_signal=False
    )

    models_data_bg = [
        {
            "label": "Athena Tight Cut",
            "bin_centers": bg_eff_athena["bin_centers"],
            "efficiencies": bg_eff_athena["efficiencies"],
            "uncertainties": bg_eff_athena["uncertainties"],
            "color": "black",
            "marker": "s",
            "is_ml": False,
            "is_reference": True,
        },
        {
            "label": f"Neural Model (Fold {best_model_info.get('fold_idx', 0)})",
            "bin_centers": bg_eff_model["bin_centers"],
            "efficiencies": bg_eff_model["efficiencies"],
            "uncertainties": bg_eff_model["uncertainties"],
            "color": "tab:red",
            "marker": "^",
            "is_ml": True,
            "is_reference": False,
        },
    ]

    plotter.plot_efficiency_vs_mu(
        models_data=models_data_bg,
        region_label=region_label,
        is_signal=False,
        output_path=args.output_dir / f"fake_rate_vs_mu_background.{args.format}",
        file_format=args.format,
    )

    try:
        npv_values = PileupAnalyzer.extract_feature(test_features, feature_names, "ph_nGoodVtx")
        vtx_corr = PileupAnalyzer.compute_vertex_correlation(mu_values, npv_values, bins=30)
        plotter.plot_vertex_correlation(
            correlation_result=vtx_corr,
            region_label=region_label,
            output_path=args.output_dir / f"vertex_correlation_npv_vs_mu.{args.format}",
            file_format=args.format,
        )
    except KeyError:
        print("Feature 'ph_nGoodVtx' not found. Skipping vertex correlation.")

    print(f"Pileup analysis complete. Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()

