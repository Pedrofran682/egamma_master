from functools import cache
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.Plotting.BasePlotter import BasePlotter
from src.core.Plotting.Context import RegionPlotContext


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


class ModelMetricsPlotter(BasePlotter):
    """Renders tripartite loss, accuracy, and SP/TPR/FPR evolution curves over epochs."""

    def __init__(self) -> None:
        """Initializes the ModelMetricsPlotter."""
        super().__init__(name="ModelMetricsPlotter")

    def plot(
        self,
        context: RegionPlotContext,
        history_data: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Plots training and validation metrics across training epochs.

        Args:
            context: RegionPlotContext with output directory settings.
            history_data: History dictionary containing loss, acc, and callback metrics.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure.
        """
        epochs = range(1, len(history_data["train_loss"]) + 1)
        train_loss = history_data["train_loss"]
        val_loss = history_data["val_loss"]
        train_acc = history_data["train_acc"]
        val_acc = history_data["val_acc"]

        callback_metrics = history_data.get("callbackMetrics", {})
        val_sp = callback_metrics.get("max_sp_val")
        val_fa = callback_metrics.get("max_sp_fa_val")
        val_pd = callback_metrics.get("max_sp_pd_val")

        fig = plt.figure(figsize=(18, 6), clear=True, num=1)

        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_loss, "o-", label="Training Loss")
        plt.plot(epochs, val_loss, "o-", label="Validation Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_acc, "o-", label="Training Accuracy")
        plt.plot(epochs, val_acc, "o-", label="Validation Accuracy")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 3)
        if val_sp is not None:
            plt.plot(epochs, val_sp, "o-", label="SP metric", color="purple")
        if val_pd is not None:
            plt.plot(epochs, val_pd, "o-", label="TPR (True Positives)", color="darkgreen")
        if val_fa is not None:
            plt.plot(epochs, val_fa, "o-", label="FPR (False Positives)", color="darkred")

        if val_sp is not None:
            best_sp_epoch_idx = np.argmax(val_sp)
            plt.axvline(
                x=epochs[best_sp_epoch_idx],
                color="blue",
                linestyle="--",
                label=f"Best SP (Epoch {epochs[best_sp_epoch_idx]})",
            )

        plt.title("Performance Metrics (SP, TPR, FPR)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"iet{context.iet}.ieta{context.ieta}_model_metrics.pdf"
        return self.save_figure(fig, context.output_dir, "model_metrics", filename)


class RocPlotter(BasePlotter):
    """Renders the Receiver Operating Characteristic (ROC) curve with AUC score."""

    def __init__(self) -> None:
        """Initializes the RocPlotter."""
        super().__init__(name="RocPlotter")

    def plot(
        self,
        context: RegionPlotContext,
        best_model_details: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Plots the ROC curve for a given model's callback details.

        Args:
            context: RegionPlotContext with kinematic region indices.
            best_model_details: Dictionary containing 'pd' (TPR), 'fa' (FPR), and 'auc_score'.
            **kwargs: Extra plotting parameters.

        Returns:
            Absolute file path of the generated PDF figure.
        """
        tpr, fpr = best_model_details["pd"], best_model_details["fa"]
        auc_score = best_model_details["auc_score"]

        fig = plt.figure(figsize=(8, 6), clear=True, num=1)
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"Model ROC Curve\n({context.iet},{context.ieta})")
        plt.legend(loc="lower right")
        plt.grid(True)

        filename = f"iet{context.iet}.ieta{context.ieta}_model_ROC.pdf"
        return self.save_figure(fig, context.output_dir, "ROC", filename)


class BoxplotSPPlotter(BasePlotter):
    """Renders boxplot and swarmplot distributions of validation metrics across CV folds."""

    def __init__(self) -> None:
        """Initializes the BoxplotSPPlotter."""
        super().__init__(name="BoxplotSPPlotter")

    def plot(
        self,
        context: RegionPlotContext,
        all_training_results: List[Dict[str, Union[str, int, object]]],
        data_type: str = "best_sp_value",
        **kwargs: Any,
    ) -> str | None:
        """Plots distribution of a chosen metric across folds.

        Args:
            context: RegionPlotContext with output directory settings.
            all_training_results: List of training result records across repeats and folds.
            data_type: Metric key to plot ('best_sp_value', 'best_fa_value', 'best_pd_value').
            **kwargs: Extra plotting parameters.

        Returns:
            Saved PDF file path, or None if results are empty.
        """
        if not all_training_results:
            return None

        labels_map = {
            "best_sp_value": ("Distribution of SP Index by Fold", "Best SP Index"),
            "best_fa_value": ("Distribution of FA by Fold", "Fake Rate"),
            "best_pd_value": ("Distribution of PD by Fold", "Efficiency"),
        }
        title, y_label = labels_map.get(data_type, (f"Distribution of {data_type}", data_type))
        df_results = pd.DataFrame(all_training_results).dropna(subset=[data_type])

        if df_results.empty:
            return None

        fig = plt.figure(figsize=(12, 7), clear=True, num=1)
        sns.boxplot(x="fold", y=data_type, data=df_results, palette="plasma", hue="fold", legend=False)
        sns.swarmplot(x="fold", y=data_type, data=df_results, palette="dark:black", alpha=0.7, size=3, hue="fold", legend=False)
        plt.title(title)
        plt.xlabel("Fold Number")
        plt.ylabel(y_label)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(
            ticks=range(int(df_results["fold"].max()) + 1),
            labels=[f"Fold {f+1}" for f in range(int(df_results["fold"].max()) + 1)],
        )

        filename = f"et{context.iet}.eta{context.ieta}.boxplot_SP.pdf"
        return self.save_figure(fig, context.output_dir, "boxplot_SP", filename)


class MetricsGridPlotter(BasePlotter):
    """Renders a 2D kinematic grid showing performance metrics (AUC, SP) per (ET, eta) pair."""

    def __init__(self) -> None:
        """Initializes the MetricsGridPlotter."""
        super().__init__(name="MetricsGridPlotter")

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
        fig, ax = plt.subplots(figsize=(10, 2))
        unique_et = df["et"].unique()
        unique_eta = df["eta"].unique()
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        for _, row in df.iterrows():
            label = f"AUC: {row['auc']:.3f}\nSP: {row['best_sp_value']:.3f}"
            ax.text(row["et"], row["eta"], label, ha="center", va="center", bbox=props, fontsize=10)

        ax.set_ylabel("$\\eta$ (eta)")
        ax.set_xlabel("$E_T$ (et)")
        ax.set_title(f"[{model_name}] Metrics by pair ($E_T$, $\\eta$)")

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
