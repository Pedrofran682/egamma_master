import pathlib
from typing import Dict, List
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.core.Plotting import BilingualPlotOrchestrator, PlotTextKey, PlotTranslator
from src.core.Plotting.Context import MetricPlotContext, RegionPlotContext
from src.core.Plotting.FeatureDistributionPlotter import FeatureDistributionPlotter
from src.core.Plotting.MetricPlotter import (
    BoxplotSPPlotter,
    MetricsGridPlotter,
    ModelMetricsPlotter,
    RocPlotter,
)
from src.core.Plotting.PileupPlotter import PileupPlotter
from src.core.Plotting.PlotManager import PlotManager
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter
from src.core.Plotting.QuadrantPlotter import QuadrantPlotter
from src.core.Plotting.RegionDistributionPlotter import RegionDistributionPlotter
from src.core.Validation.EfficiencyPlotter import EfficiencyPlotter
from src.core.Validation.PileupAnalyzer import PileupAnalyzer
from src.core.Validation.QuadrantAnalyzer import (
    QuadrantMetrics,
    RegionalQuadrantResult,
    StrategyMetrics,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def test_translator_all_keys_have_en_and_pt() -> None:
    assert len(PlotTranslator.TRANSLATIONS) > 0
    assert set(PlotTextKey) == set(PlotTranslator.TRANSLATIONS.keys())
    for key, entries in PlotTranslator.TRANSLATIONS.items():
        assert isinstance(key, PlotTextKey)
        assert "en" in entries and entries["en"].strip(), f"Missing 'en' for key '{key}'"
        assert "pt" in entries and entries["pt"].strip(), f"Missing 'pt' for key '{key}'"


def test_translator_lookup_formatting_and_errors() -> None:
    en_title = PlotTranslator.translate(PlotTextKey.METRIC_LOSS_TITLE, language="en")
    pt_title = PlotTranslator.translate(PlotTextKey.METRIC_LOSS_TITLE, language="pt")
    assert en_title == "Loss Curves"
    assert pt_title == "Curvas de Perda"
    assert en_title != pt_title
    assert PlotTranslator.translate("metric.loss_title", language="pt") == pt_title

    en_formatted = PlotTranslator.translate(PlotTextKey.ROC_LEGEND, language="en", auc=0.9812)
    pt_formatted = PlotTranslator.translate(PlotTextKey.ROC_LEGEND, language="pt", auc=0.9812)
    assert "0.9812" in en_formatted
    assert "0.9812" in pt_formatted

    en_best = PlotTranslator.translate(PlotTextKey.METRIC_BEST_SP_EPOCH, language="en", epoch=5)
    pt_best = PlotTranslator.translate(PlotTextKey.METRIC_BEST_SP_EPOCH, language="pt", epoch=5)
    assert "Epoch 5" in en_best
    assert "Época 5" in pt_best

    with pytest.raises(ValueError):
        PlotTranslator.validate_language("es")

    with pytest.raises(ValueError):
        PlotTranslator.translate(PlotTextKey.METRIC_LOSS_TITLE, language="fr")

    with pytest.raises(KeyError):
        PlotTranslator.translate("nonexistent.translation.key", language="en")


def test_bilingual_orchestrator_resolution_and_execution(tmp_path: pathlib.Path) -> None:
    orch_en = BilingualPlotOrchestrator(language="en")
    assert orch_en.languages == ["en"]
    assert not orch_en.is_multi_language
    assert orch_en.resolve_output_dir(tmp_path, "en", False) == tmp_path

    orch_pt = BilingualPlotOrchestrator(language="pt")
    assert orch_pt.languages == ["pt"]
    assert not orch_pt.is_multi_language
    assert orch_pt.resolve_output_dir(tmp_path, "pt", False) == tmp_path

    orch_both = BilingualPlotOrchestrator(language="both")
    assert orch_both.languages == ["en", "pt"]
    assert orch_both.is_multi_language
    assert orch_both.resolve_output_dir(tmp_path, "en", True) == tmp_path / "en"
    assert orch_both.resolve_output_dir(tmp_path, "pt", True) == tmp_path / "pt"

    with pytest.raises(ValueError):
        BilingualPlotOrchestrator(language="de")

    with pytest.raises(ValueError):
        BilingualPlotOrchestrator.resolve_languages("invalid")

    visited: Dict[str, pathlib.Path] = {}

    def _task(lang: str, target_dir: pathlib.Path) -> str:
        visited[lang] = target_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / "marker.txt"
        out_file.write_text(lang)
        return str(out_file)

    results = orch_both.orchestrate(tmp_path, _task)
    assert list(results.keys()) == ["en", "pt"]
    assert visited["en"] == tmp_path / "en"
    assert visited["pt"] == tmp_path / "pt"
    assert (tmp_path / "en" / "marker.txt").read_text() == "en"
    assert (tmp_path / "pt" / "marker.txt").read_text() == "pt"


def test_metric_plotters_bilingual_generation_and_text(tmp_path: pathlib.Path) -> None:
    df_region = pd.DataFrame(
        [
            {
                "fold": 0,
                "best_sp_value": 0.90,
                "history": {
                    "train_loss": [0.4, 0.25],
                    "val_loss": [0.45, 0.3],
                    "train_acc": [0.80, 0.89],
                    "val_acc": [0.78, 0.87],
                    "callbackMetrics": {
                        "pd": [0.85, 0.92],
                        "fa": [0.15, 0.08],
                        "sp": [0.85, 0.90],
                        "auc_score": 0.9512,
                        "best_knee_point": 1,
                    },
                },
            }
        ]
    )

    captured_titles: Dict[str, List[str]] = {"en": [], "pt": []}
    original_set_title = matplotlib.axes.Axes.set_title
    original_title = plt.title

    def _run_for_lang(lang: str, target_dir: pathlib.Path) -> List[str]:
        ctx = RegionPlotContext(
            iet=1,
            ieta=2,
            output_dir=str(target_dir),
            data=np.array([]),
            target=np.array([]),
            language=lang,
        )
        mctx = MetricPlotContext(
            region_context=ctx,
            history_data=df_region.iloc[0]["history"],
            callback_metrics=df_region.iloc[0]["history"]["callbackMetrics"],
            all_training_results=df_region,
        )
        p1 = ModelMetricsPlotter(language=lang).plot_metric(mctx)
        p2 = RocPlotter(language=lang).plot_metric(mctx)
        p3 = BoxplotSPPlotter(language=lang).plot_metric(mctx)

        grid_df = pd.DataFrame(
            [
                {
                    "et": 1,
                    "eta": 2,
                    "percentage": 1.0,
                    "model_name": "ModelV1",
                    "auc": 0.9512,
                    "best_sp_value": 0.90,
                }
            ]
        )
        grid_dir = target_dir / "grid"
        grid_ctx = RegionPlotContext(
            iet=1,
            ieta=2,
            output_dir=str(grid_dir),
            data=None,
            target=None,
            language=lang,
        )
        p4 = MetricsGridPlotter(language=lang).plot(grid_ctx, df=grid_df, model_name="ModelV1")
        return [p for p in [p1, p2, p3, p4] if p is not None]

    current_lang = "en"

    def _spy_set_title(self_ax, label, *args, **kwargs):
        captured_titles[current_lang].append(str(label))
        return original_set_title(self_ax, label, *args, **kwargs)

    def _spy_plt_title(label, *args, **kwargs):
        captured_titles[current_lang].append(str(label))
        return original_title(label, *args, **kwargs)

    orchestrator = BilingualPlotOrchestrator(language="both")
    for lang in orchestrator.languages:
        current_lang = lang
        resolved_dir = orchestrator.resolve_output_dir(tmp_path, lang, True)
        with (
            patch.object(matplotlib.axes.Axes, "set_title", _spy_set_title),
            patch("src.core.Plotting.MetricPlotter.plt.title", side_effect=_spy_plt_title),
        ):
            paths = _run_for_lang(lang, resolved_dir)
        assert len(paths) == 4
        for path_str in paths:
            out_file = pathlib.Path(path_str)
            assert out_file.exists() and out_file.stat().st_size > 0

    assert "Loss Curves" in captured_titles["en"]
    assert "Curvas de Perda" in captured_titles["pt"]
    assert any("Model ROC Curve" in t for t in captured_titles["en"])
    assert any("Curva ROC do Modelo" in t for t in captured_titles["pt"])
    assert any("[ModelV1] Metrics by pair" in t for t in captured_titles["en"])
    assert any("[ModelV1] Métricas por par" in t for t in captured_titles["pt"])


def test_profile_plotter_bilingual(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(7)
    data = rng.uniform(0.1, 1.0, size=(30, 10))
    target = np.array([1] * 15 + [0] * 15)

    captured_xlabels: Dict[str, List[str]] = {"en": [], "pt": []}
    original_set_xlabel = matplotlib.axes.Axes.set_xlabel

    for lang in ("en", "pt"):
        lang_dir = tmp_path / lang
        ctx = RegionPlotContext(
            iet=1,
            ieta=1,
            output_dir=str(lang_dir),
            data=data,
            target=target,
            language=lang,
        )

        def _spy_set_xlabel(self_ax, label, *args, **kwargs):
            captured_xlabels[lang].append(str(label))
            return original_set_xlabel(self_ax, label, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "set_xlabel", _spy_set_xlabel):
            saved = ProfileMeanEnergyPlotter(language=lang).plot(ctx, x_rings=data)

        assert pathlib.Path(saved).exists()
        assert pathlib.Path(saved).stat().st_size > 0

    assert "Calorimeter Ring Index" in captured_xlabels["en"]
    assert "Índice do Anel no Calorímetro" in captured_xlabels["pt"]


def test_region_distribution_plotter_bilingual(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame(
        [
            {
                "et": 0,
                "eta": 0,
                "et_range": "15-20",
                "eta_range": "0.00-0.80",
                "background_count": 100,
                "signal_count": 120,
                "total_events": 220,
                "signal_fraction": 120 / 220,
            }
        ]
    )

    orchestrator = BilingualPlotOrchestrator(language="both")

    def _run_lang(lang: str, resolved_dir: pathlib.Path) -> Dict[str, str]:
        plotter = RegionDistributionPlotter(
            et_intervals={0: "15-20"},
            eta_intervals={0: "0.00-0.80"},
            language=lang,
        )
        return plotter.plot(
            region_distribution_dataframe=df,
            output_dir=resolved_dir,
            file_format="png",
            subfolder="",
        )

    outputs = orchestrator.orchestrate(tmp_path, _run_lang)
    for lang in ("en", "pt"):
        assert "composite_grid" in outputs[lang]
        for path_str in outputs[lang].values():
            p = pathlib.Path(path_str)
            assert p.exists() and p.stat().st_size > 0
            assert p.parent == tmp_path / lang


def test_quadrant_plotter_bilingual(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(11)
    n_samples = 120
    labels = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    preds1 = labels.copy()
    preds2 = labels.copy()
    preds1[:10] = 1 - preds1[:10]
    preds2[5:15] = 1 - preds2[5:15]

    result = RegionalQuadrantResult(
        iet=1,
        ieta=1,
        strategy1_name="ModelA",
        strategy2_name="AthenaLoose",
        metrics_strategy1=StrategyMetrics.from_predictions(preds1, labels),
        metrics_strategy2=StrategyMetrics.from_predictions(preds2, labels),
        overall_metrics=QuadrantMetrics(
            both_correct=95,
            model1_only_correct=10,
            model2_only_correct=10,
            both_wrong=5,
            total_events=n_samples,
        ),
        signal_metrics=QuadrantMetrics(
            both_correct=48,
            model1_only_correct=5,
            model2_only_correct=5,
            both_wrong=2,
            total_events=60,
        ),
        background_metrics=QuadrantMetrics(
            both_correct=47,
            model1_only_correct=5,
            model2_only_correct=5,
            both_wrong=3,
            total_events=60,
        ),
        preds_strategy1=preds1,
        preds_strategy2=preds2,
        labels=labels,
        threshold_strategy1=0.5,
        threshold_strategy2=0.5,
        showershapes={"reta": rng.normal(0.94, 0.03, size=n_samples)},
        et=rng.uniform(20.0, 40.0, size=n_samples),
        eta=rng.uniform(0.1, 1.3, size=n_samples),
    )

    orchestrator = BilingualPlotOrchestrator(language="both")

    def _run_lang(lang: str, resolved_dir: pathlib.Path) -> Dict[str, List[str]]:
        return QuadrantPlotter(language=lang).plot(
            results=[result],
            output_dir=resolved_dir,
            file_format="png",
        )

    outputs = orchestrator.orchestrate(tmp_path, _run_lang)
    for lang in ("en", "pt"):
        total = sum(len(paths) for paths in outputs[lang].values())
        assert total > 0
        for path_list in outputs[lang].values():
            for path_str in path_list:
                p = pathlib.Path(path_str)
                assert p.exists() and p.stat().st_size > 0
                assert str(tmp_path / lang) in str(p)


def test_feature_distribution_plotter_bilingual(tmp_path: pathlib.Path) -> None:
    centers = np.linspace(0.85, 1.0, 8)
    true_counts = np.array([10, 20, 35, 50, 60, 40, 20, 10], dtype=float)
    feature_data = {
        "bin_centers": centers,
        "true_counts": true_counts,
        "display_title": r"$R_{\eta}$",
        "models": [
            {
                "name": "ModelV1",
                "label": "ModelV1",
                "counts": true_counts * 0.9,
                "errors": np.sqrt(true_counts * 0.9),
                "color": "tab:blue",
                "marker": "o",
                "is_ml": True,
            }
        ],
    }

    en_out = tmp_path / "en_feat.png"
    pt_out = tmp_path / "pt_feat.png"
    fig_en = FeatureDistributionPlotter(language="en").plot_single_feature(
        feature_data=feature_data,
        feature_name="reta",
        region_key="iet1.ieta1",
        is_signal=False,
        output_path=en_out,
        plot_format="png",
    )
    fig_pt = FeatureDistributionPlotter(language="pt").plot_single_feature(
        feature_data=feature_data,
        feature_name="reta",
        region_key="iet1.ieta1",
        is_signal=False,
        output_path=pt_out,
        plot_format="png",
    )

    assert en_out.exists() and en_out.stat().st_size > 0
    assert pt_out.exists() and pt_out.stat().st_size > 0
    assert "Jet Leak in Signal" in fig_en.axes[0].get_title()
    assert "Vazamento de Jatos no Sinal" in fig_pt.axes[0].get_title()
    assert "Events / Bin" in fig_en.axes[0].get_ylabel()
    assert "Eventos de" in fig_pt.axes[0].get_ylabel()

    fig_grid_en = FeatureDistributionPlotter(language="en").plot_overview_grid(
        plot_data={"reta": feature_data},
        features_to_plot=["reta"],
        region_key="iet1.ieta1",
        is_signal=False,
    )
    fig_grid_pt = FeatureDistributionPlotter(language="pt").plot_overview_grid(
        plot_data={"reta": feature_data},
        features_to_plot=["reta"],
        region_key="iet1.ieta1",
        is_signal=False,
    )
    assert "Holdout Jet Leakage in Signal Selection" in fig_grid_en._suptitle.get_text()
    assert "Seleção de Vazamento de Jatos no Sinal (Holdout)" in fig_grid_pt._suptitle.get_text()


def test_pileup_plotter_bilingual(tmp_path: pathlib.Path) -> None:
    centers = np.linspace(20.0, 50.0, 5)
    models_data = [
        {
            "label": "ModelV1",
            "bin_centers": centers,
            "efficiencies": np.array([0.95, 0.94, 0.93, 0.92, 0.90]),
            "uncertainties": np.array([0.01, 0.01, 0.01, 0.01, 0.02]),
            "color": "tab:blue",
            "marker": "o",
            "is_ml": True,
            "is_reference": True,
        }
    ]
    corr_res = PileupAnalyzer.compute_vertex_correlation(
        mu_values=centers * 1.5,
        npv_values=centers * 0.8,
        bins=8,
    )

    fig_en = PileupPlotter(language="en").plot_efficiency_vs_mu(
        models_data=models_data,
        region_label="iet1.ieta1",
        output_path=tmp_path / "en_eff_mu.png",
        file_format="png",
    )
    fig_pt = PileupPlotter(language="pt").plot_efficiency_vs_mu(
        models_data=models_data,
        region_label="iet1.ieta1",
        output_path=tmp_path / "pt_eff_mu.png",
        file_format="png",
    )

    assert (tmp_path / "en_eff_mu.png").exists()
    assert (tmp_path / "pt_eff_mu.png").exists()
    assert "Signal Efficiency ($P_D$)" in fig_en.axes[0].get_title()
    assert "Eficiência de Sinal ($P_D$)" in fig_pt.axes[0].get_title()

    fig_corr_en = PileupPlotter(language="en").plot_vertex_correlation(
        correlation_result=corr_res,
        region_label="iet1.ieta1",
        output_path=tmp_path / "en_corr.png",
        file_format="png",
    )
    fig_corr_pt = PileupPlotter(language="pt").plot_vertex_correlation(
        correlation_result=corr_res,
        region_label="iet1.ieta1",
        output_path=tmp_path / "pt_corr.png",
        file_format="png",
    )
    assert "Primary Vertices" in fig_corr_en.axes[0].get_title()
    assert "Vértices Primários" in fig_corr_pt.axes[0].get_title()


def test_efficiency_plotter_bilingual_and_inheritance(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(17)
    n_events = 60
    x_holdout = rng.uniform(15.0, 50.0, size=(n_events, 2))
    y_holdout = np.array([1] * 30 + [0] * 30)
    preds = y_holdout.copy()

    captured_ylabels: Dict[str, List[str]] = {"en": [], "pt": []}
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    for lang in ("en", "pt"):
        plotter = EfficiencyPlotter(et_index=0, eta_index=1, language=lang)
        assert plotter.language == lang
        out_path = tmp_path / f"eff_{lang}.png"

        def _spy_set_ylabel(self_ax, label, *args, **kwargs):
            captured_ylabels[lang].append(str(label))
            return original_set_ylabel(self_ax, label, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "set_ylabel", _spy_set_ylabel):
            plotter.plot_efficiency_vs_feature(
                feature_values=x_holdout[:, 0],
                y_holdout=y_holdout,
                preds=preds,
                file_path=out_path,
                feature_name="ET [GeV]",
            )

        assert out_path.exists() and out_path.stat().st_size > 0

    assert captured_ylabels["en"] == ["Detection Efficiency (TPR)"]
    assert captured_ylabels["pt"] == ["Eficiência de Detecção (TPR)"]

    both_dir = tmp_path / "manager_both"
    PlotManager(language="both").run_region_plots(
        RegionPlotContext(
            iet=0,
            ieta=0,
            output_dir=str(both_dir),
            data=rng.uniform(0.1, 1.0, size=(20, 10)),
            target=np.array([1] * 10 + [0] * 10),
        )
    )
    assert (both_dir / "en").exists()
    assert (both_dir / "pt").exists()
