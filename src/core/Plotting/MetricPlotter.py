from functools import cache
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.Interfaces.BasePlotter import BaseMetricPlotter, BasePlotter
from src.core.Plotting.Context import MetricPlotContext, RegionPlotContext
from src.core.Plotting.Localization import PlotTextKey


@cache
def get_et_axis() -> List[str]:
    """Provides human-readable labels for transverse energy (ET) bin intervals.

    Returns:
        List of formatted interval strings.
    """
    return [
        "0: 25->30",
        "1: 30->35",
        "2: 35->40",
        "3: 40->45",
        "4: 45->50",
        "5: 50->60",
        "6: 60->80",
        "7: 80->100",
    ]


@cache
def get_eta_axis() -> List[str]:
    """Provides human-readable labels for pseudorapidity (eta) bin intervals.

    Returns:
        List of formatted interval strings.
    """
    return [
        "0: 0.00->0.60",
        "1: 0.60->0.80",
        "2: 0.80->1.15",
        "3: 1.15->1.37",
        "4: 1.37->1.52",
        "5: 1.52->1.81",
        "6: 1.81->2.01",
        "7: 2.01->2.37",
        "8: 2.37->2.47",
    ]


class ModelMetricsPlotter(BaseMetricPlotter):
    """Renders tripartite loss, accuracy, and SP/TPR/FPR evolution curves over epochs."""

    def __init__(self, language: str = "en") -> None:
        """Initializes the ModelMetricsPlotter.

        Args:
            language: Target language code ('en' or 'pt').
        """
        super().__init__(name="ModelMetricsPlotter", language=language)

    def plot_metric(self, context: MetricPlotContext) -> str | None:
        """Plots training and validation metrics using MetricPlotContext."""
        if not context.history_data:
            return None
        return self._render_plot(context.region_context, context.history_data)

    def plot(
        self,
        context: RegionPlotContext,
        history_data: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Plots training and validation metrics across training epochs.

        Args:
            context: RegionPlotContext with output directory settings.
            history_data: History dictionary containing loss, acc, and callback metrics.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure, or None if missing data.
        """
        if not history_data:
            return None
        return self._render_plot(context, history_data)

    def _render_plot(self, context: RegionPlotContext, history_data: Dict[str, Any]) -> str:
        lang = self._resolve_language(context)
        epochs = range(1, len(history_data["train_loss"]) + 1)
        train_loss = history_data["train_loss"]
        val_loss = history_data["val_loss"]
        train_acc = history_data["train_acc"]
        val_acc = history_data["val_acc"]

        callback_metrics = history_data.get("callbackMetrics", {})
        val_sp = callback_metrics.get("max_sp_val")
        val_fa = callback_metrics.get("max_sp_fa_val")
        val_pd = callback_metrics.get("max_sp_pd_val")

        fig, (ax_loss, ax_acc, ax_perf) = plt.subplots(1, 3, figsize=(18, 6))

        ax_loss.plot(epochs, train_loss, "o-", label=self.tr(PlotTextKey.METRIC_TRAIN_LOSS, language=lang))
        ax_loss.plot(epochs, val_loss, "o-", label=self.tr(PlotTextKey.METRIC_VAL_LOSS, language=lang))
        ax_loss.set_title(self.tr(PlotTextKey.METRIC_LOSS_TITLE, language=lang))
        ax_loss.set_xlabel(self.tr(PlotTextKey.METRIC_EPOCH, language=lang))
        ax_loss.set_ylabel(self.tr(PlotTextKey.METRIC_LOSS, language=lang))
        ax_loss.grid(True)
        ax_loss.legend()

        ax_acc.plot(epochs, train_acc, "o-", label=self.tr(PlotTextKey.METRIC_TRAIN_ACC, language=lang))
        ax_acc.plot(epochs, val_acc, "o-", label=self.tr(PlotTextKey.METRIC_VAL_ACC, language=lang))
        ax_acc.set_title(self.tr(PlotTextKey.METRIC_ACC_TITLE, language=lang))
        ax_acc.set_xlabel(self.tr(PlotTextKey.METRIC_EPOCH, language=lang))
        ax_acc.set_ylabel(self.tr(PlotTextKey.METRIC_ACC, language=lang))
        ax_acc.grid(True)
        ax_acc.legend()

        has_val_curve = False
        if val_sp is not None:
            ax_perf.plot(epochs, val_sp, "o-", label=self.tr(PlotTextKey.METRIC_SP_LABEL, language=lang), color="purple")
            has_val_curve = True
        if val_pd is not None:
            ax_perf.plot(epochs, val_pd, "o-", label=self.tr(PlotTextKey.METRIC_TPR_LABEL, language=lang), color="darkgreen")
            has_val_curve = True
        if val_fa is not None:
            ax_perf.plot(epochs, val_fa, "o-", label=self.tr(PlotTextKey.METRIC_FPR_LABEL, language=lang), color="darkred")
            has_val_curve = True

        if val_sp is not None:
            best_sp_epoch_idx = np.argmax(val_sp)
            ax_perf.axvline(
                x=epochs[best_sp_epoch_idx],
                color="blue",
                linestyle="--",
                label=self.tr(PlotTextKey.METRIC_BEST_SP_EPOCH, language=lang, epoch=epochs[best_sp_epoch_idx]),
            )

        ax_perf.set_title(self.tr(PlotTextKey.METRIC_PERF_TITLE, language=lang))
        ax_perf.set_xlabel(self.tr(PlotTextKey.METRIC_EPOCH, language=lang))
        ax_perf.set_ylabel(self.tr(PlotTextKey.METRIC_VALUE, language=lang))
        ax_perf.grid(True)
        if has_val_curve:
            ax_perf.legend()
        fig.tight_layout()

        filename = f"iet{context.iet}.ieta{context.ieta}_model_metrics.pdf"
        return self.save_figure(fig, context.output_dir, "model_metrics", filename)


class RocPlotter(BaseMetricPlotter):
    """Renders the Receiver Operating Characteristic (ROC) curve with AUC score."""

    def __init__(self, language: str = "en") -> None:
        """Initializes the RocPlotter.

        Args:
            language: Target language code ('en' or 'pt').
        """
        super().__init__(name="RocPlotter", language=language)

    def plot_metric(self, context: MetricPlotContext) -> str | None:
        """Plots ROC curve using MetricPlotContext."""
        details = context.callback_metrics
        if details is None and context.history_data:
            details = context.history_data.get("callbackMetrics")
        if not details or "pd" not in details or "fa" not in details:
            return None
        return self._render_plot(context.region_context, details)

    def plot(
        self,
        context: RegionPlotContext,
        best_model_details: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Plots the ROC curve for a given model's callback details.

        Args:
            context: RegionPlotContext with kinematic region indices.
            best_model_details: Dictionary containing 'pd' (TPR), 'fa' (FPR), and 'auc_score'.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure, or None if details are missing.
        """
        if not best_model_details or "pd" not in best_model_details or "fa" not in best_model_details:
            return None
        return self._render_plot(context, best_model_details)

    def _render_plot(self, context: RegionPlotContext, best_model_details: Dict[str, Any]) -> str:
        lang = self._resolve_language(context)
        tpr, fpr = best_model_details["pd"], best_model_details["fa"]
        auc_score = best_model_details.get("auc_score", 0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="blue", lw=2, label=self.tr(PlotTextKey.ROC_LEGEND, language=lang, auc=auc_score))
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel(self.tr(PlotTextKey.ROC_XLABEL, language=lang))
        ax.set_ylabel(self.tr(PlotTextKey.ROC_YLABEL, language=lang))
        ax.set_title(self.tr(PlotTextKey.ROC_TITLE, language=lang, iet=context.iet, ieta=context.ieta))
        ax.legend(loc="lower right")
        ax.grid(True)

        filename = f"iet{context.iet}.ieta{context.ieta}_model_ROC.pdf"
        return self.save_figure(fig, context.output_dir, "ROC", filename)


class BoxplotSPPlotter(BaseMetricPlotter):
    """Renders boxplot and swarmplot distributions of validation metrics across CV folds."""

    def __init__(self, language: str = "en") -> None:
        """Initializes the BoxplotSPPlotter.

        Args:
            language: Target language code ('en' or 'pt').
        """
        super().__init__(name="BoxplotSPPlotter", language=language)

    def plot_metric(self, context: MetricPlotContext) -> str | None:
        """Plots distribution of metrics using MetricPlotContext."""
        return self._render_plot(context.region_context, context.all_training_results)

    def plot(
        self,
        context: RegionPlotContext,
        all_training_results: Union[List[Dict[str, Union[str, int, object]]], pd.DataFrame, None],
        data_type: str = "best_sp_value",
        **kwargs: Any,
    ) -> str | None:
        """Plots distribution of a chosen metric across folds.

        Args:
            context: RegionPlotContext with output directory settings.
            all_training_results: Training result records (DataFrame or list of dicts).
            data_type: Metric key to plot ('best_sp_value', 'best_fa_value', 'best_pd_value').
            **kwargs: Extra plotting parameters.

        Returns:
            Saved PDF file path, or None if results are empty.
        """
        return self._render_plot(context, all_training_results, data_type=data_type)

    def _render_plot(
        self,
        context: RegionPlotContext,
        all_training_results: Union[List[Dict[str, Union[str, int, object]]], pd.DataFrame, None],
        data_type: str = "best_sp_value",
    ) -> str | None:
        if all_training_results is None:
            return None

        if isinstance(all_training_results, pd.DataFrame):
            df_results = all_training_results
        else:
            df_results = pd.DataFrame(all_training_results)

        if df_results.empty or data_type not in df_results.columns:
            return None

        df_results = df_results.dropna(subset=[data_type])
        if df_results.empty:
            return None

        lang = self._resolve_language(context)
        key_map = {
            "best_sp_value": (PlotTextKey.BOXPLOT_SP_TITLE, PlotTextKey.BOXPLOT_SP_YLABEL),
            "best_fa_value": (PlotTextKey.BOXPLOT_FA_TITLE, PlotTextKey.BOXPLOT_FA_YLABEL),
            "best_pd_value": (PlotTextKey.BOXPLOT_PD_TITLE, PlotTextKey.BOXPLOT_PD_YLABEL),
        }
        if data_type in key_map:
            title_key, ylabel_key = key_map[data_type]
            title = self.tr(title_key, language=lang)
            y_label = self.tr(ylabel_key, language=lang)
        else:
            title = self.tr(PlotTextKey.BOXPLOT_DEFAULT_TITLE, language=lang, data_type=data_type)
            y_label = data_type

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(x="fold", y=data_type, data=df_results, palette="plasma", hue="fold", legend=False, ax=ax)
        sns.swarmplot(x="fold", y=data_type, data=df_results, palette="dark:black", alpha=0.7, size=3, hue="fold", legend=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(self.tr(PlotTextKey.BOXPLOT_XLABEL, language=lang))
        ax.set_ylabel(y_label)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        max_fold = int(df_results["fold"].max())
        ax.set_xticks(
            ticks=range(max_fold + 1),
            labels=[self.tr(PlotTextKey.BOXPLOT_FOLD_TICK, language=lang, fold=f + 1) for f in range(max_fold + 1)],
        )

        filename = f"et{context.iet}.eta{context.ieta}.boxplot_SP.pdf"
        return self.save_figure(fig, context.output_dir, "boxplot_SP", filename)


class MetricsGridPlotter(BasePlotter):
    """Renders a 2D kinematic grid showing performance metrics (AUC, SP) per (ET, eta) pair."""

    def __init__(self, language: str = "en") -> None:
        """Initializes the MetricsGridPlotter.

        Args:
            language: Target language code ('en' or 'pt').
        """
        super().__init__(name="MetricsGridPlotter", language=language)

    def plot(
        self,
        context: RegionPlotContext,
        df: pd.DataFrame,
        model_name: str,
        **kwargs: Any,
    ) -> str:
        """Plots summary metrics arranged in a grid across kinematic bins.

        Args:
            context: RegionPlotContext with output directory settings.
            df: DataFrame containing aggregated metrics per (et, eta) bin.
            model_name: Model identifier used for labeling.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PNG figure.
        """
        lang = self._resolve_language(context)
        fig, ax = plt.subplots(figsize=(10, 2))
        unique_et = df["et"].unique()
        unique_eta = df["eta"].unique()
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        for _, row in df.iterrows():
            label = self.tr(PlotTextKey.GRID_CELL_LABEL, language=lang, auc=row["auc"], sp=row["best_sp_value"])
            ax.text(row["et"], row["eta"], label, ha="center", va="center", bbox=props, fontsize=10)

        ax.set_ylabel(self.tr(PlotTextKey.GRID_YLABEL, language=lang))
        ax.set_xlabel(self.tr(PlotTextKey.GRID_XLABEL, language=lang))
        ax.set_title(self.tr(PlotTextKey.GRID_TITLE, language=lang, model_name=model_name))

        margin_x = 0.5
        margin_y = 1.0
        ax.set_xlim(min(unique_eta) - margin_x, max(unique_eta) + margin_x + 7)
        ax.set_ylim(min(unique_et) - margin_y, max(unique_et) + margin_y)

        ax.set_yticks(sorted(unique_eta), get_eta_axis()[: len(unique_eta)])
        ax.set_xticks(sorted(unique_et), get_et_axis()[: len(unique_et)])
        ax.grid(True, linestyle="--", alpha=0.3)

        filename = f"{model_name}_plot_metrics_grid.png"
        return self.save_figure(
            fig,
            context.output_dir,
            "plot_metrics_grid",
            filename,
            file_format="png",
            transparent=False,
        )
