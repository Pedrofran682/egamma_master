import os
import pathlib
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.core.Plotting.BasePlotter import BasePlotter
from src.utils import create_folder


class RegionDistributionPlotter(BasePlotter):
    """Plotter generating 2D heatmap grids of dataset event distributions across ET and eta regions."""

    def __init__(self) -> None:
        """Initializes the RegionDistributionPlotter."""
        super().__init__(name="RegionDistributionPlotter")

    def plot(
        self,
        region_distribution_dataframe: pd.DataFrame,
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
        subfolder: str = "DataDistribution",
        **kwargs: object,
    ) -> Dict[str, str]:
        """Renders and saves 2D heatmap plots for background, signal, total counts, and signal ratio.

        Args:
            region_distribution_dataframe: DataFrame with regional count and ratio metrics.
            output_dir: Base directory path where figures will be saved.
            file_format: Graphic format ('pdf', 'png').
            subfolder: Subfolder name for output files.
            **kwargs: Extra plotting parameters.

        Returns:
            Dictionary mapping plot names to saved file paths.
        """
        output_directory_path = create_folder(subfolder, str(output_dir))
        saved_figure_paths: Dict[str, str] = {}

        composite_grid_path = self.plot_composite_grid(
            region_distribution_dataframe=region_distribution_dataframe,
            output_dir=output_directory_path,
            file_format=file_format,
        )
        saved_figure_paths["composite_grid"] = composite_grid_path

        metric_configurations = [
            ("background_count", "Background (Class 0) Event Count", "Blues", ".0f"),
            ("signal_count", "Signal (Class 1) Event Count", "Greens", ".0f"),
            ("total_events", "Total Event Count", "Purples", ".0f"),
            ("signal_fraction", "Signal Fraction (Class 1 / Total)", "YlGnBu", ".2%"),
        ]

        for metric_column, subplot_title, colormap_name, format_specifier in metric_configurations:
            figure, subplot_axis = plt.subplots(figsize=(8, 6))
            self._render_single_heatmap(
                subplot_axis=subplot_axis,
                region_distribution_dataframe=region_distribution_dataframe,
                metric_column=metric_column,
                subplot_title=subplot_title,
                colormap_name=colormap_name,
                format_specifier=format_specifier,
            )
            plot_filename = f"distribution_{metric_column}.{file_format}"
            destination_path = os.path.join(str(output_directory_path), plot_filename)
            figure.savefig(destination_path, format=file_format, dpi=300, bbox_inches="tight")
            plt.close(figure)
            saved_figure_paths[metric_column] = destination_path

        return saved_figure_paths

    def plot_composite_grid(
        self,
        region_distribution_dataframe: pd.DataFrame,
        output_dir: str | pathlib.Path,
        file_format: str = "pdf",
    ) -> str:
        """Renders and saves a 2x2 composite heatmap grid of all metrics.

        Args:
            region_distribution_dataframe: DataFrame containing region distribution metrics.
            output_dir: Directory where the figure will be saved.
            file_format: Output image format.

        Returns:
            Saved figure path string.
        """
        composite_figure, subplot_axes = plt.subplots(2, 2, figsize=(16, 12))
        metric_configurations = [
            ("background_count", "Background (Class 0) Counts", "Blues", ".0f"),
            ("signal_count", "Signal (Class 1) Counts", "Greens", ".0f"),
            ("total_events", "Total Counts", "Purples", ".0f"),
            ("signal_fraction", "Signal Fraction (Class 1 / Total)", "YlGnBu", ".2%"),
        ]

        for current_axis, (metric_column, subplot_title, colormap_name, format_specifier) in zip(
            subplot_axes.flat, metric_configurations
        ):
            self._render_single_heatmap(
                subplot_axis=current_axis,
                region_distribution_dataframe=region_distribution_dataframe,
                metric_column=metric_column,
                subplot_title=subplot_title,
                colormap_name=colormap_name,
                format_specifier=format_specifier,
            )

        composite_figure.suptitle(
            "Data Distribution by Class across Kinematic Regions",
            fontsize=16,
            weight="bold",
        )
        composite_figure.tight_layout()

        grid_filename = f"data_distribution_2d_grid.{file_format}"
        saved_grid_path = os.path.join(str(output_dir), grid_filename)
        composite_figure.savefig(saved_grid_path, format=file_format, dpi=300, bbox_inches="tight")
        plt.close(composite_figure)
        return saved_grid_path

    def _render_single_heatmap(
        self,
        subplot_axis: plt.Axes,
        region_distribution_dataframe: pd.DataFrame,
        metric_column: str,
        subplot_title: str,
        colormap_name: str,
        format_specifier: str,
    ) -> None:
        """Renders an individual heatmap on a matplotlib Axes.

        Args:
            subplot_axis: Matplotlib Axes to draw on.
            region_distribution_dataframe: Distribution summary DataFrame.
            metric_column: Column name to visualize.
            subplot_title: Title of the heatmap subplot.
            colormap_name: Seaborn/Matplotlib colormap name.
            format_specifier: String format specifier for cell annotations.
        """
        region_matrix_pivot = region_distribution_dataframe.pivot(
            index="et", columns="eta", values=metric_column
        )
        region_matrix_pivot = region_matrix_pivot.sort_index(ascending=False)

        sns.heatmap(
            region_matrix_pivot,
            ax=subplot_axis,
            annot=True,
            fmt=format_specifier,
            cmap=colormap_name,
            cbar=True,
            linewidths=0.5,
            linecolor="white",
        )
        subplot_axis.set_title(subplot_title, fontsize=12, pad=10)
        subplot_axis.set_xlabel(r"$\eta$ Bin", fontsize=11)
        subplot_axis.set_ylabel(r"$E_T$ Bin", fontsize=11)
