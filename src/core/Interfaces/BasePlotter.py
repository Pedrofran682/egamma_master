import os
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.core.Plotting.Context import RegionPlotContext
    from src.core.Plotting.Localization import PlotTextKey
from src.utils import create_folder


class BasePlotter(ABC):
    """Abstract base class for all plotting strategy components.

    Attributes:
        name: Identifier name for the plotter instance.
        language: Active language code ('en' or 'pt') for plot text localization.
    """

    def __init__(self, name: Optional[str] = None, language: str = "en") -> None:
        """Initializes the plotter with a name and localization language.

        Args:
            name: Descriptive name of the plotter (defaults to class name).
            language: Target language code ('en' or 'pt').
        """
        self.name: str = name if name is not None else self.__class__.__name__
        self.language: str = self._get_translator().validate_language(language)

    @staticmethod
    def _get_translator() -> Any:
        from src.core.Plotting.Localization import PlotTranslator

        return PlotTranslator

    def _resolve_language(self, context: Any = None, language: Optional[str] = None) -> str:
        """Resolves the active language from explicit argument, context, or plotter default.

        Args:
            context: Optional plot context object carrying a language attribute.
            language: Optional explicit language override.

        Returns:
            Validated single-language code ('en' or 'pt').
        """
        translator = self._get_translator()
        if language is not None:
            return translator.validate_language(language)
        if context is not None:
            ctx_lang = getattr(context, "language", None)
            if ctx_lang is None and hasattr(context, "region_context"):
                ctx_lang = getattr(context.region_context, "language", None)
            if ctx_lang is not None and ctx_lang != "en":
                return translator.validate_language(ctx_lang)
        return translator.validate_language(self.language)

    def tr(
        self,
        key: Union["PlotTextKey", str],
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Translates a localization key using the provided or configured language.

        Args:
            key: Translation lookup key in PlotTranslator.TRANSLATIONS.
            language: Optional language override ('en' or 'pt').
            **kwargs: Optional format placeholders interpolated into the template.

        Returns:
            Localized and formatted text string.
        """
        translator = self._get_translator()
        target_lang = (
            translator.validate_language(language)
            if language is not None
            else translator.validate_language(self.language)
        )
        return translator.translate(key, language=target_lang, **kwargs)

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
        transparent: bool = False,
        show: bool = False,
    ) -> str:
        """Saves a Matplotlib figure safely and frees graphic resources from memory.

        Args:
            fig: Matplotlib Figure instance to save.
            folder_path: Target root directory.
            subfolder: Subfolder topic name (e.g. 'RingsMeanProfiles').
            filename: Output filename including extension.
            file_format: Graphic format ('pdf', 'png').
            dpi: Dots per inch resolution.
            transparent: Whether figure background should be transparent (default: False).
            show: Whether to display the figure inline before closing it (default: False).

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
                facecolor="white",
                transparent=transparent,
                bbox_inches="tight",
            )
            if show:
                plt.show()
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
