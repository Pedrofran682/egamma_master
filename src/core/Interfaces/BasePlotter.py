import os
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.core.Plotting.Context import RegionPlotContext
from src.utils import create_folder


class BasePlotter(ABC):
    """Abstract base class for all plotting strategy components.

    Attributes:
        name: Identifier name for the plotter instance.
    """

    def __init__(self, name: str) -> None:
        """Initializes the plotter with a name.

        Args:
            name: Descriptive name of the plotter.
        """
        self.name: str = name

    @abstractmethod
    def plot(self, context: Any, **kwargs: Any) -> Any:
        """Generates figure and executes rendering.

        Args:
            context: RegionPlotContext containing kinematic data and parameters.
            **kwargs: Plotter-specific arguments.
        """
        pass

    def plot_metric(self, context: Any) -> Any:
        """Optional metric plotting hook for plotters consuming MetricPlotContext.

        Args:
            context: MetricPlotContext instance.
        """
        return None

    @staticmethod
    def save_figure(
        fig: plt.Figure,
        folder_path: str | pathlib.Path,
        subfolder: str,
        filename: str,
        file_format: str = "pdf",
        dpi: int = 300,
        transparent: bool = True,
    ) -> str:
        """Saves a Matplotlib figure safely and frees graphic resources from memory.

        Args:
            fig: Matplotlib Figure instance to save.
            folder_path: Target root directory.
            subfolder: Subfolder topic name (e.g. 'RingsMeanProfiles').
            filename: Output filename including extension.
            file_format: Graphic format ('pdf', 'png').
            dpi: Dots per inch resolution.
            transparent: Whether figure background should be transparent.

        Returns:
            Absolute file path where the figure was saved.

        Raises:
            Exception: If an error occurs during file writing.
        """
        try:
            path = create_folder(subfolder, str(folder_path))
            save_path = os.path.join(str(path), filename)
            fig.savefig(
                save_path,
                format=file_format,
                dpi=dpi,
                transparent=transparent,
                bbox_inches="tight",
            )
            plt.close(fig)
            return save_path
        except Exception as e:
            plt.close(fig)
            raise e


class BaseMetricPlotter(BasePlotter, ABC):
    """Abstract base class for metric and evaluation plotters.

    Attributes:
        name: Descriptive identifier for the metric plotter.
    """

    @abstractmethod
    def plot_metric(self, context: Any) -> Any:
        """Renders metric figure using standardized MetricPlotContext.

        Args:
            context: MetricPlotContext containing region data, history, and results.
        """
        pass
