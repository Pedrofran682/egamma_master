import logging
from typing import Any, List
import numpy as np
import pandas as pd

from src.core.Plotting.BasePlotter import BasePlotter
from src.core.Plotting.Context import RegionPlotContext
from src.core.Plotting.MetricPlotter import BoxplotSPPlotter, ModelMetricsPlotter, RocPlotter
from src.core.Plotting.ProfilePlotter import ProfileMeanEnergyPlotter

log = logging.getLogger()


class PlotManager:
    """Manages registration and execution of visual plotters for evaluations.

    Attributes:
        plotters: Registry list of active BasePlotter instances.
    """

    def __init__(self, include_default_plotters: bool = True) -> None:
        """Initializes PlotManager with optional default plotters.

        Args:
            include_default_plotters: Whether to automatically register profile and metric plotters.
        """
        self.plotters: List[BasePlotter] = []
        if include_default_plotters:
            self.register_plotter(ProfileMeanEnergyPlotter())
            self.register_plotter(ModelMetricsPlotter())
            self.register_plotter(RocPlotter())
            self.register_plotter(BoxplotSPPlotter())

    def register_plotter(self, plotter: BasePlotter) -> None:
        """Registers a new plotter strategy into the execution pipeline.

        Args:
            plotter: BasePlotter instance to append.
        """
        self.plotters.append(plotter)

    def run_region_plots(self, context: RegionPlotContext, **kwargs: Any) -> None:
        """Executes all registered plotters sequentially for a given kinematic context.

        Args:
            context: RegionPlotContext holding data, targets, and indices.
            **kwargs: Extra arguments passed to each plotter's plot method.
        """
        for plotter in self.plotters:
            try:
                plotter.plot(context, **kwargs)
            except Exception as e:
                log.error(f"Error running plotter {plotter.name}: {e}")

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

        context = RegionPlotContext(
            iet=iet,
            ieta=ieta,
            output_dir=output_dir,
            data=np.array([]),
            target=np.array([]),
        )

        for plotter in self.plotters:
            if isinstance(plotter, ModelMetricsPlotter):
                if "history" in best_run and best_run["history"]:
                    plotter.plot(context, history_data=best_run["history"])
            elif isinstance(plotter, RocPlotter):
                if (
                    "history" in best_run
                    and isinstance(best_run["history"], dict)
                    and "callbackMetrics" in best_run["history"]
                ):
                    plotter.plot(
                        context,
                        best_model_details=best_run["history"]["callbackMetrics"],
                    )
            elif isinstance(plotter, BoxplotSPPlotter):
                plotter.plot(context, all_training_results=df_region)

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
