from dataclasses import replace
import logging
from pathlib import Path
from typing import Any, Callable, List
import numpy as np
import pandas as pd

from src.core.Interfaces.BasePlotter import BasePlotter
from src.core.Plotting.Context import MetricPlotContext, RegionPlotContext
from src.core.Plotting.Localization import BilingualPlotOrchestrator
from src.core.Plotting.MetricPlotter import BoxplotSPPlotter, ModelMetricsPlotter, RocPlotter
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter

log = logging.getLogger(__name__)


class PlotManager:
    """Manages registration and execution of visual plotters for evaluations.

    Attributes:
        plotters: Registry list of active BasePlotter instances.
        orchestrator: BilingualPlotOrchestrator controlling target languages and folders.
    """

    def __init__(
        self, include_default_plotters: bool = True, language: str = "en"
    ) -> None:
        """Initializes PlotManager with optional default plotters.

        Args:
            include_default_plotters: Whether to automatically register profile and metric plotters.
            language: Target localization language ('en', 'pt', or 'both').
        """
        self.orchestrator: BilingualPlotOrchestrator = BilingualPlotOrchestrator(
            language=language
        )
        default_lang = self.orchestrator.languages[0]
        self.plotters: List[BasePlotter] = []
        if include_default_plotters:
            self.register_plotter(ProfileMeanEnergyPlotter(language=default_lang))
            self.register_plotter(ModelMetricsPlotter(language=default_lang))
            self.register_plotter(RocPlotter(language=default_lang))
            self.register_plotter(BoxplotSPPlotter(language=default_lang))

    def register_plotter(self, plotter: BasePlotter) -> None:
        """Registers a new plotter strategy into the execution pipeline.

        Args:
            plotter: BasePlotter instance to append.
        """
        self.plotters.append(plotter)

    @staticmethod
    def _run_with_language(
        plotter: BasePlotter, lang: str, action: Callable[[], None]
    ) -> None:
        """Executes a plotter callback with its language temporarily set to lang."""
        has_lang = hasattr(plotter, "language")
        prev_lang = getattr(plotter, "language", None)
        if has_lang:
            plotter.language = lang
        try:
            action()
        finally:
            if has_lang and prev_lang is not None:
                plotter.language = prev_lang

    def run_region_plots(self, context: RegionPlotContext, **kwargs: Any) -> None:
        """Executes all registered plotters sequentially for a given kinematic context.

        Args:
            context: RegionPlotContext holding data, targets, and indices.
            **kwargs: Extra arguments passed to each plotter's plot method.
        """
        def _render_language(lang: str, resolved_dir: Path) -> None:
            active_lang = (
                lang
                if (self.orchestrator.is_multi_language or self.orchestrator.language != "en")
                else context.language
            )
            target_dir = (
                str(resolved_dir)
                if self.orchestrator.is_multi_language
                else context.output_dir
            )
            lang_context = replace(context, output_dir=target_dir, language=active_lang)
            for plotter in self.plotters:
                try:
                    self._run_with_language(
                        plotter,
                        active_lang,
                        lambda p=plotter, ctx=lang_context: p.plot(ctx, **kwargs),
                    )
                except Exception as e:
                    log.error(f"Error running plotter {plotter.name}: {e}")

        self.orchestrator.orchestrate(context.output_dir, _render_language)

    def run_metrics_for_region(
        self,
        df_region: pd.DataFrame,
        output_dir: str,
        iet: int,
        ieta: int,
    ) -> bool:
        """Selects the best model run by highest SP value and renders region metric plots.

        Args:
            df_region: Consolidated DataFrame with fold and repeat records for the region.
            output_dir: Target output directory for charts.
            iet: Transverse energy bin index.
            ieta: Pseudorapidity bin index.

        Returns:
            True if plots were generated, False if skipped due to missing/empty data.
        """
        if (
            df_region is None
            or df_region.empty
            or "best_sp_value" not in df_region.columns
            or df_region["best_sp_value"].dropna().empty
        ):
            log.warning(f"No valid metric data found for region et={iet}, eta={ieta}. Skipping.")
            return False

        valid_df = df_region.dropna(subset=["best_sp_value"])
        best_run = valid_df.loc[valid_df["best_sp_value"].idxmax()]

        history = best_run.get("history") if "history" in best_run else None
        callback_metrics = (
            history.get("callbackMetrics")
            if isinstance(history, dict)
            else None
        )

        def _render_language(lang: str, resolved_dir: Path) -> None:
            target_dir = (
                str(resolved_dir)
                if self.orchestrator.is_multi_language
                else output_dir
            )
            context = RegionPlotContext(
                iet=iet,
                ieta=ieta,
                output_dir=target_dir,
                data=np.array([]),
                target=np.array([]),
                language=lang,
            )
            metric_context = MetricPlotContext(
                region_context=context,
                history_data=history if isinstance(history, dict) else None,
                callback_metrics=callback_metrics,
                all_training_results=df_region,
            )
            for plotter in self.plotters:
                try:
                    self._run_with_language(
                        plotter,
                        lang,
                        lambda p=plotter, mctx=metric_context: p.plot_metric(mctx),
                    )
                except Exception as e:
                    log.error(f"Error running metric plotter {plotter.name}: {e}")

        self.orchestrator.orchestrate(output_dir, _render_language)
        return True

    def run_metrics_from_file(
        self, results_data_path: str, output_dir: str, iet: int, ieta: int
    ) -> bool:
        """Loads a results pickle file, selects the best model run, and triggers metric plots.

        Args:
            results_data_path: File path of the pickled results DataFrame.
            output_dir: Target output directory for charts.
            iet: Transverse energy bin index.
            ieta: Pseudorapidity bin index.

        Returns:
            True if plots were generated, False if skipped due to missing/empty data.
        """
        try:
            data = pd.read_pickle(results_data_path)
            df = pd.DataFrame(data)
        except Exception as e:
            log.warning(f"Failed to load result file {results_data_path}: {e}. Skipping.")
            return False

        return self.run_metrics_for_region(df, output_dir, iet, ieta)
