from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Literal, Tuple, Union

SupportedLanguage = Literal["en", "pt"]
LanguageOption = Literal["en", "pt", "both"]


class PlotTextKey(str, Enum):
    """Enumeration of all localized plot text keys."""

    METRIC_TRAIN_LOSS = "metric.train_loss"
    METRIC_VAL_LOSS = "metric.val_loss"
    METRIC_LOSS_TITLE = "metric.loss_title"
    METRIC_EPOCH = "metric.epoch"
    METRIC_LOSS = "metric.loss"
    METRIC_TRAIN_ACC = "metric.train_acc"
    METRIC_VAL_ACC = "metric.val_acc"
    METRIC_ACC_TITLE = "metric.acc_title"
    METRIC_ACC = "metric.acc"
    METRIC_SP_LABEL = "metric.sp_label"
    METRIC_TPR_LABEL = "metric.tpr_label"
    METRIC_FPR_LABEL = "metric.fpr_label"
    METRIC_BEST_SP_EPOCH = "metric.best_sp_epoch"
    METRIC_PERF_TITLE = "metric.perf_title"
    METRIC_VALUE = "metric.value"
    ROC_LEGEND = "roc.legend"
    ROC_XLABEL = "roc.xlabel"
    ROC_YLABEL = "roc.ylabel"
    ROC_TITLE = "roc.title"
    BOXPLOT_SP_TITLE = "boxplot.sp_title"
    BOXPLOT_SP_YLABEL = "boxplot.sp_ylabel"
    BOXPLOT_FA_TITLE = "boxplot.fa_title"
    BOXPLOT_FA_YLABEL = "boxplot.fa_ylabel"
    BOXPLOT_PD_TITLE = "boxplot.pd_title"
    BOXPLOT_PD_YLABEL = "boxplot.pd_ylabel"
    BOXPLOT_DEFAULT_TITLE = "boxplot.default_title"
    BOXPLOT_XLABEL = "boxplot.xlabel"
    BOXPLOT_FOLD_TICK = "boxplot.fold_tick"
    GRID_XLABEL = "grid.xlabel"
    GRID_YLABEL = "grid.ylabel"
    GRID_TITLE = "grid.title"
    GRID_CELL_LABEL = "grid.cell_label"
    PROFILE_TITLE = "profile.title"
    PROFILE_SIGNAL_LABEL = "profile.signal_label"
    PROFILE_BACKGROUND_LABEL = "profile.background_label"
    PROFILE_XLABEL = "profile.xlabel"
    PROFILE_YLABEL = "profile.ylabel"
    PROFILE_REGIONAL_TITLE = "profile.regional_title"
    PROFILE_GLOBAL_TITLE = "profile.global_title"
    DATASET_DIST_REGIONAL_TITLE = "dataset_dist.regional_title"
    DATASET_DIST_GLOBAL_TITLE = "dataset_dist.global_title"
    DATASET_DIST_YLABEL_DENSITY = "dataset_dist.ylabel_density"
    DATASET_DIST_YLABEL_COUNTS = "dataset_dist.ylabel_counts"
    DATASET_DIST_LOG_SUFFIX = "dataset_dist.log_suffix"
    DATASET_YIELD_XLABEL = "dataset_yield.xlabel"
    DATASET_YIELD_YLABEL = "dataset_yield.ylabel"
    DATASET_YIELD_TITLE = "dataset_yield.title"
    CONV_TITLE = "conv.title"
    SALIENCY_SIGNAL_MEAN = "saliency.signal_mean"
    SALIENCY_SIGNAL_STD = "saliency.signal_std"
    SALIENCY_BACKGROUND_MEAN = "saliency.background_mean"
    SALIENCY_BACKGROUND_STD = "saliency.background_std"
    SALIENCY_TITLE = "saliency.title"
    SALIENCY_XLABEL = "saliency.xlabel"
    SALIENCY_YLABEL = "saliency.ylabel"
    REGION_DIST_BACKGROUND_COUNT_TITLE = "region_dist.background_count_title"
    REGION_DIST_SIGNAL_COUNT_TITLE = "region_dist.signal_count_title"
    REGION_DIST_TOTAL_EVENTS_TITLE = "region_dist.total_events_title"
    REGION_DIST_SIGNAL_FRACTION_TITLE = "region_dist.signal_fraction_title"
    REGION_DIST_BACKGROUND_COUNT_SHORT = "region_dist.background_count_short"
    REGION_DIST_SIGNAL_COUNT_SHORT = "region_dist.signal_count_short"
    REGION_DIST_TOTAL_EVENTS_SHORT = "region_dist.total_events_short"
    REGION_DIST_GRID_SUPTITLE = "region_dist.grid_suptitle"
    REGION_DIST_XLABEL = "region_dist.xlabel"
    REGION_DIST_YLABEL = "region_dist.ylabel"
    QUADRANT_VAR_TRIG_L2_CALO_WETA2 = "quadrant.var.trig_L2_calo_weta2"
    QUADRANT_VAR_TRIG_L2_CALO_WSTOT = "quadrant.var.trig_L2_calo_wstot"
    QUADRANT_VAR_TRIG_L2_CALO_FRACS1 = "quadrant.var.trig_L2_calo_fracs1"
    QUADRANT_GLOBAL_REGION = "quadrant.global_region"
    QUADRANT_BIN_FALLBACK = "quadrant.bin_fallback"
    QUADRANT_REGION_LABEL = "quadrant.region_label"
    QUADRANT_BOTH_CORRECT = "quadrant.both_correct"
    QUADRANT_STRATEGY_ONLY = "quadrant.strategy_only"
    QUADRANT_BOTH_WRONG = "quadrant.both_wrong"
    QUADRANT_HIST_SUPTITLE = "quadrant.hist_suptitle"
    QUADRANT_SIGNAL_TITLE = "quadrant.signal_title"
    QUADRANT_BACKGROUND_TITLE = "quadrant.background_title"
    QUADRANT_TRUE_SIGNAL = "quadrant.true_signal"
    QUADRANT_TRUE_BACKGROUND = "quadrant.true_background"
    QUADRANT_EVENTS_YLABEL = "quadrant.events_ylabel"
    QUADRANT_RATIO_YLABEL = "quadrant.ratio_ylabel"
    QUADRANT_VAR_ET = "quadrant.var_et"
    QUADRANT_VAR_ETA = "quadrant.var_eta"
    QUADRANT_SIGNAL_EFF_YLABEL = "quadrant.signal_eff_ylabel"
    QUADRANT_SIGNAL_EFF_VS_TITLE = "quadrant.signal_eff_vs_title"
    QUADRANT_BG_EFF_YLABEL = "quadrant.bg_eff_ylabel"
    QUADRANT_BG_EFF_VS_TITLE = "quadrant.bg_eff_vs_title"
    QUADRANT_EFF_SUPTITLE = "quadrant.eff_suptitle"
    QUADRANT_SUMMARY_YLABEL = "quadrant.summary_ylabel"
    QUADRANT_SUMMARY_TITLE = "quadrant.summary_title"
    QUADRANT_CLASS_EFF_SIG_TITLE = "quadrant.class_eff_sig_title"
    QUADRANT_CLASS_EFF_BG_TITLE = "quadrant.class_eff_bg_title"
    QUADRANT_CLASS_EFF_SUPTITLE = "quadrant.class_eff_suptitle"
    FEATURE_DIST_TRUE_SIGNAL = "feature_dist.true_signal"
    FEATURE_DIST_TRUE_BACKGROUND = "feature_dist.true_background"
    FEATURE_DIST_PREFIX_SIGNAL = "feature_dist.prefix_signal"
    FEATURE_DIST_PREFIX_BACKGROUND = "feature_dist.prefix_background"
    FEATURE_DIST_SUBPLOT_TITLE = "feature_dist.subplot_title"
    FEATURE_DIST_EVENTS_YLABEL = "feature_dist.events_ylabel"
    FEATURE_DIST_RATIO_YLABEL = "feature_dist.ratio_ylabel"
    FEATURE_DIST_OVERVIEW_HEADLINE_SIGNAL = "feature_dist.overview_headline_signal"
    FEATURE_DIST_OVERVIEW_HEADLINE_BACKGROUND = "feature_dist.overview_headline_background"
    FEATURE_DIST_OVERVIEW_SUPTITLE = "feature_dist.overview_suptitle"
    FEATURE_DIST_REGION_SUFFIX = "feature_dist.region_suffix"
    PILEUP_SIGNAL_EFFICIENCY = "pileup.signal_efficiency"
    PILEUP_BACKGROUND_FAKE_RATE = "pileup.background_fake_rate"
    PILEUP_VS_MU_TITLE = "pileup.vs_mu_title"
    PILEUP_RATIO_TO = "pileup.ratio_to"
    PILEUP_AVGMU_XLABEL = "pileup.avgmu_xlabel"
    PILEUP_MEAN_OBSERVABLE_YLABEL = "pileup.mean_observable_ylabel"
    PILEUP_EVENTS_PER_BIN = "pileup.events_per_bin"
    PILEUP_MEAN_NPV_LABEL = "pileup.mean_npv_label"
    PILEUP_FIT_NPV_LABEL = "pileup.fit_npv_label"
    PILEUP_VERTEX_TITLE = "pileup.vertex_title"
    PILEUP_NPV_YLABEL = "pileup.npv_ylabel"
    PILEUP_SIGNAL_EVENTS_PER_BIN = "pileup.signal_events_per_bin"
    PILEUP_FIT_THRESHOLD_LABEL = "pileup.fit_threshold_label"
    PILEUP_THRESHOLD_CALIB_TITLE = "pileup.threshold_calib_title"
    PILEUP_DISCRIMINANT_YLABEL = "pileup.discriminant_ylabel"
    EFFICIENCY_DEFAULT_FEATURE = "efficiency.default_feature"
    EFFICIENCY_DEFAULT_LABEL = "efficiency.default_label"
    EFFICIENCY_YLABEL = "efficiency.ylabel"
    EFFICIENCY_TITLE_VS = "efficiency.title_vs"
    EFFICIENCY_ET_STANDARD_FEATURE = "efficiency.et_standard_feature"
    EFFICIENCY_ET_PROPOSED_FEATURE = "efficiency.et_proposed_feature"
    EFFICIENCY_ETA_STANDARD_FEATURE = "efficiency.eta_standard_feature"
    EFFICIENCY_ETA_PROPOSED_FEATURE = "efficiency.eta_proposed_feature"
    EFFICIENCY_LABEL_STANDARD_CUT = "efficiency.label_standard_cut"
    EFFICIENCY_LABEL_PROPOSED_CUT = "efficiency.label_proposed_cut"
    PEAK_PROFILE_TITLE = "peak.profile_title"
    PEAK_INTRA_BAR_TITLE = "peak.intra_bar_title"
    PEAK_GLOBAL_BAR_TITLE = "peak.global_bar_title"
    PEAK_BAR_SUPTITLE = "peak.bar_suptitle"
    PEAK_SUBDETECTOR_XLABEL = "peak.subdetector_xlabel"
    PEAK_INTRA_YLABEL = "peak.intra_ylabel"
    PEAK_GLOBAL_YLABEL = "peak.global_ylabel"
    PEAK_HEATMAP_TITLE = "peak.heatmap_title"
    PEAK_REGION_XLABEL = "peak.region_xlabel"
    PEAK_REGION_YLABEL = "peak.region_ylabel"
    PEAK_HIGHLIGHT_LABEL = "peak.highlight_label"
    PEAK_TOTAL_SUBDET_LABEL = "peak.total_subdet_label"


class PlotTranslator:
    """Provides bilingual (English and Portuguese) text translations for all plotters.

    Attributes:
        SUPPORTED_LANGUAGES: Tuple of supported single-language codes.
        TRANSLATIONS: Mapping from translation keys to language-specific templates.
    """

    SUPPORTED_LANGUAGES: ClassVar[Tuple[str, ...]] = ("en", "pt")

    TRANSLATIONS: ClassVar[Dict[PlotTextKey, Dict[str, str]]] = {
        PlotTextKey.METRIC_TRAIN_LOSS: {
            "en": "Training Loss",
            "pt": "Perda de Treinamento",
        },
        PlotTextKey.METRIC_VAL_LOSS: {
            "en": "Validation Loss",
            "pt": "Perda de Validação",
        },
        PlotTextKey.METRIC_LOSS_TITLE: {
            "en": "Loss Curves",
            "pt": "Curvas de Perda",
        },
        PlotTextKey.METRIC_EPOCH: {
            "en": "Epoch",
            "pt": "Época",
        },
        PlotTextKey.METRIC_LOSS: {
            "en": "Loss",
            "pt": "Perda",
        },
        PlotTextKey.METRIC_TRAIN_ACC: {
            "en": "Training Accuracy",
            "pt": "Acurácia de Treinamento",
        },
        PlotTextKey.METRIC_VAL_ACC: {
            "en": "Validation Accuracy",
            "pt": "Acurácia de Validação",
        },
        PlotTextKey.METRIC_ACC_TITLE: {
            "en": "Accuracy Curves",
            "pt": "Curvas de Acurácia",
        },
        PlotTextKey.METRIC_ACC: {
            "en": "Accuracy",
            "pt": "Acurácia",
        },
        PlotTextKey.METRIC_SP_LABEL: {
            "en": "SP metric",
            "pt": "Métrica SP",
        },
        PlotTextKey.METRIC_TPR_LABEL: {
            "en": "TPR (True Positives)",
            "pt": "TPR (Verdadeiros Positivos)",
        },
        PlotTextKey.METRIC_FPR_LABEL: {
            "en": "FPR (False Positives)",
            "pt": "FPR (Falsos Positivos)",
        },
        PlotTextKey.METRIC_BEST_SP_EPOCH: {
            "en": "Best SP (Epoch {epoch})",
            "pt": "Melhor SP (Época {epoch})",
        },
        PlotTextKey.METRIC_PERF_TITLE: {
            "en": "Performance Metrics (SP, TPR, FPR)",
            "pt": "Métricas de Desempenho (SP, TPR, FPR)",
        },
        PlotTextKey.METRIC_VALUE: {
            "en": "Metric value",
            "pt": "Valor da métrica",
        },
        PlotTextKey.ROC_LEGEND: {
            "en": "ROC (AUC = {auc:.4f})",
            "pt": "ROC (AUC = {auc:.4f})",
        },
        PlotTextKey.ROC_XLABEL: {
            "en": "False Positive Rate (FPR)",
            "pt": "Taxa de Falsos Positivos (FPR)",
        },
        PlotTextKey.ROC_YLABEL: {
            "en": "True Positive Rate (TPR)",
            "pt": "Taxa de Verdadeiros Positivos (TPR)",
        },
        PlotTextKey.ROC_TITLE: {
            "en": "Model ROC Curve\n({iet},{ieta})",
            "pt": "Curva ROC do Modelo\n({iet},{ieta})",
        },
        PlotTextKey.BOXPLOT_SP_TITLE: {
            "en": "Distribution of SP Index by Fold",
            "pt": "Distribuição do Índice SP por Fold",
        },
        PlotTextKey.BOXPLOT_SP_YLABEL: {
            "en": "Best SP Index",
            "pt": "Melhor Índice SP",
        },
        PlotTextKey.BOXPLOT_FA_TITLE: {
            "en": "Distribution of FA by Fold",
            "pt": "Distribuição de FA por Fold",
        },
        PlotTextKey.BOXPLOT_FA_YLABEL: {
            "en": "Fake Rate",
            "pt": "Taxa de Falsos Alarmes",
        },
        PlotTextKey.BOXPLOT_PD_TITLE: {
            "en": "Distribution of PD by Fold",
            "pt": "Distribuição de PD por Fold",
        },
        PlotTextKey.BOXPLOT_PD_YLABEL: {
            "en": "Efficiency",
            "pt": "Eficiência",
        },
        PlotTextKey.BOXPLOT_DEFAULT_TITLE: {
            "en": "Distribution of {data_type}",
            "pt": "Distribuição de {data_type}",
        },
        PlotTextKey.BOXPLOT_XLABEL: {
            "en": "Fold Number",
            "pt": "Número do Fold",
        },
        PlotTextKey.BOXPLOT_FOLD_TICK: {
            "en": "Fold {fold}",
            "pt": "Fold {fold}",
        },
        PlotTextKey.GRID_XLABEL: {
            "en": "$E_T$ (et)",
            "pt": "$E_T$ (et)",
        },
        PlotTextKey.GRID_YLABEL: {
            "en": "$\\eta$ (eta)",
            "pt": "$\\eta$ (eta)",
        },
        PlotTextKey.GRID_TITLE: {
            "en": "[{model_name}] Metrics by pair ($E_T$, $\\eta$)",
            "pt": "[{model_name}] Métricas por par ($E_T$, $\\eta$)",
        },
        PlotTextKey.GRID_CELL_LABEL: {
            "en": "AUC: {auc:.3f}\nSP: {sp:.3f}",
            "pt": "AUC: {auc:.3f}\nSP: {sp:.3f}",
        },
        PlotTextKey.PROFILE_TITLE: {
            "en": "Average Energy Profile in the Rings - NeuralRinger",
            "pt": "Perfil Médio de Energia nos Anéis - NeuralRinger",
        },
        PlotTextKey.PROFILE_SIGNAL_LABEL: {
            "en": "Photons",
            "pt": "Fótons",
        },
        PlotTextKey.PROFILE_BACKGROUND_LABEL: {
            "en": "Hadronic Jets",
            "pt": "Jatos Hadrônicos",
        },
        PlotTextKey.PROFILE_XLABEL: {
            "en": "Calorimeter Ring Index",
            "pt": "Índice do Anel no Calorímetro",
        },
        PlotTextKey.PROFILE_YLABEL: {
            "en": "Normalized Energy",
            "pt": "Energia Normalizada",
        },
        PlotTextKey.PROFILE_REGIONAL_TITLE: {
            "en": "Average Energy Profile ({n_rings} Rings) - Region {region}",
            "pt": "Perfil Médio de Energia ({n_rings} Anéis) - Região {region}",
        },
        PlotTextKey.PROFILE_GLOBAL_TITLE: {
            "en": "Global Average Energy Profile ({n_rings} Rings) - All Processed Regions ({n_regions} regions)",
            "pt": "Perfil Médio Global de Energia ({n_rings} Anéis) - Todas as Regiões Processadas ({n_regions} regiões)",
        },
        PlotTextKey.DATASET_DIST_REGIONAL_TITLE: {
            "en": "Distribution of {variable} (Region {region})",
            "pt": "Distribuição de {variable} (Região {region})",
        },
        PlotTextKey.DATASET_DIST_GLOBAL_TITLE: {
            "en": "Global Distribution of {variable} - All Processed Regions ({n_regions} regions)",
            "pt": "Distribuição Global de {variable} - Todas as Regiões Processadas ({n_regions} regiões)",
        },
        PlotTextKey.DATASET_DIST_YLABEL_DENSITY: {
            "en": "Density",
            "pt": "Densidade",
        },
        PlotTextKey.DATASET_DIST_YLABEL_COUNTS: {
            "en": "Counts",
            "pt": "Contagens",
        },
        PlotTextKey.DATASET_DIST_LOG_SUFFIX: {
            "en": " (log scale)",
            "pt": " (escala log)",
        },
        PlotTextKey.DATASET_YIELD_XLABEL: {
            "en": "Calorimeter Kinematic Region (ET, ETA)",
            "pt": "Região Cinemática do Calorímetro (ET, ETA)",
        },
        PlotTextKey.DATASET_YIELD_YLABEL: {
            "en": "Event Count",
            "pt": "Contagem de Eventos",
        },
        PlotTextKey.DATASET_YIELD_TITLE: {
            "en": "Event Yield Distribution across Calorimeter Regions",
            "pt": "Distribuição de Eventos nas Regiões do Calorímetro",
        },
        PlotTextKey.CONV_TITLE: {
            "en": "Feature Extraction - Conv layer output",
            "pt": "Extração de Características - Saída da camada Conv",
        },
        PlotTextKey.SALIENCY_SIGNAL_MEAN: {
            "en": "Signal (mean)",
            "pt": "Sinal (média)",
        },
        PlotTextKey.SALIENCY_SIGNAL_STD: {
            "en": "Signal ±1σ",
            "pt": "Sinal ±1σ",
        },
        PlotTextKey.SALIENCY_BACKGROUND_MEAN: {
            "en": "Background (mean)",
            "pt": "Fundo (média)",
        },
        PlotTextKey.SALIENCY_BACKGROUND_STD: {
            "en": "Background ±1σ",
            "pt": "Fundo ±1σ",
        },
        PlotTextKey.SALIENCY_TITLE: {
            "en": "Comparison of Normalized Mean Saliency Profile\n({n_samples} samples per class)\n({iet},{ieta})",
            "pt": "Comparação do Perfil Médio de Saliência Normalizado\n({n_samples} amostras por classe)\n({iet},{ieta})",
        },
        PlotTextKey.SALIENCY_XLABEL: {
            "en": "Input Position - Ring Index",
            "pt": "Posição de Entrada - Índice do Anel",
        },
        PlotTextKey.SALIENCY_YLABEL: {
            "en": "Normalized Importance (Saliency)",
            "pt": "Importância Normalizada (Saliência)",
        },
        PlotTextKey.REGION_DIST_BACKGROUND_COUNT_TITLE: {
            "en": "Background (Class 0) Event Count",
            "pt": "Contagem de Eventos de Fundo (Classe 0)",
        },
        PlotTextKey.REGION_DIST_SIGNAL_COUNT_TITLE: {
            "en": "Signal (Class 1) Event Count",
            "pt": "Contagem de Eventos de Sinal (Classe 1)",
        },
        PlotTextKey.REGION_DIST_TOTAL_EVENTS_TITLE: {
            "en": "Total Event Count",
            "pt": "Contagem Total de Eventos",
        },
        PlotTextKey.REGION_DIST_SIGNAL_FRACTION_TITLE: {
            "en": "Signal Fraction (Class 1 / Total)",
            "pt": "Fração de Sinal (Classe 1 / Total)",
        },
        PlotTextKey.REGION_DIST_BACKGROUND_COUNT_SHORT: {
            "en": "Background (Class 0) Counts",
            "pt": "Contagens de Fundo (Classe 0)",
        },
        PlotTextKey.REGION_DIST_SIGNAL_COUNT_SHORT: {
            "en": "Signal (Class 1) Counts",
            "pt": "Contagens de Sinal (Classe 1)",
        },
        PlotTextKey.REGION_DIST_TOTAL_EVENTS_SHORT: {
            "en": "Total Counts",
            "pt": "Contagens Totais",
        },
        PlotTextKey.REGION_DIST_GRID_SUPTITLE: {
            "en": "Data Distribution by Class across Kinematic Regions",
            "pt": "Distribuição dos Dados por Classe nas Regiões Cinemáticas",
        },
        PlotTextKey.REGION_DIST_XLABEL: {
            "en": r"$|\eta|$ Interval",
            "pt": r"Intervalo de $|\eta|$",
        },
        PlotTextKey.REGION_DIST_YLABEL: {
            "en": r"$E_T$ Interval",
            "pt": r"Intervalo de $E_T$",
        },
        PlotTextKey.QUADRANT_VAR_TRIG_L2_CALO_WETA2: {
            "en": r"L2 $w_{\eta 2}$ (Lateral width in EM2)",
            "pt": r"L2 $w_{\eta 2}$ (Largura lateral em EM2)",
        },
        PlotTextKey.QUADRANT_VAR_TRIG_L2_CALO_WSTOT: {
            "en": r"L2 $w_{\text{stot}}$ (Total width in strips)",
            "pt": r"L2 $w_{\text{stot}}$ (Largura total nas tiras)",
        },
        PlotTextKey.QUADRANT_VAR_TRIG_L2_CALO_FRACS1: {
            "en": r"L2 $f_{1}$ (Energy fraction in EM1)",
            "pt": r"L2 $f_{1}$ (Fração de energia em EM1)",
        },
        PlotTextKey.QUADRANT_GLOBAL_REGION: {
            "en": "Global (All Regions)",
            "pt": "Global (Todas as Regiões)",
        },
        PlotTextKey.QUADRANT_BIN_FALLBACK: {
            "en": "Bin {bin_idx}",
            "pt": "Faixa {bin_idx}",
        },
        PlotTextKey.QUADRANT_REGION_LABEL: {
            "en": "$E_T$ {et_desc}, $\\eta$ {eta_desc} (iet{iet}.ieta{ieta})",
            "pt": "$E_T$ {et_desc}, $\\eta$ {eta_desc} (iet{iet}.ieta{ieta})",
        },
        PlotTextKey.QUADRANT_BOTH_CORRECT: {
            "en": "Both Correct",
            "pt": "Ambos Corretos",
        },
        PlotTextKey.QUADRANT_STRATEGY_ONLY: {
            "en": "{strategy} Only",
            "pt": "Apenas {strategy}",
        },
        PlotTextKey.QUADRANT_BOTH_WRONG: {
            "en": "Both Wrong",
            "pt": "Ambos Errados",
        },
        PlotTextKey.QUADRANT_HIST_SUPTITLE: {
            "en": "{var_display} — Quadrant Breakdown\n{region_str} ({s1} vs {s2})",
            "pt": "{var_display} — Decomposição por Quadrantes\n{region_str} ({s1} vs {s2})",
        },
        PlotTextKey.QUADRANT_SIGNAL_TITLE: {
            "en": "Signal (Photons)",
            "pt": "Sinal (Fótons)",
        },
        PlotTextKey.QUADRANT_BACKGROUND_TITLE: {
            "en": "Background (Fakes)",
            "pt": "Fundo (Falsos)",
        },
        PlotTextKey.QUADRANT_TRUE_SIGNAL: {
            "en": "True Signal",
            "pt": "Sinal Verdadeiro",
        },
        PlotTextKey.QUADRANT_TRUE_BACKGROUND: {
            "en": "True Background",
            "pt": "Fundo Verdadeiro",
        },
        PlotTextKey.QUADRANT_EVENTS_YLABEL: {
            "en": "Events",
            "pt": "Eventos",
        },
        PlotTextKey.QUADRANT_RATIO_YLABEL: {
            "en": r"$N_{\mathrm{cat}} / N_{\mathrm{true}}$",
            "pt": r"$N_{\mathrm{cat}} / N_{\mathrm{true}}$",
        },
        PlotTextKey.QUADRANT_VAR_ET: {
            "en": r"Transverse Energy $E_T$ [GeV] ($E_T < 150$ GeV)",
            "pt": r"Energia Transversa $E_T$ [GeV] ($E_T < 150$ GeV)",
        },
        PlotTextKey.QUADRANT_VAR_ETA: {
            "en": r"Pseudorapidity $\eta$",
            "pt": r"Pseudorrapidez $\eta$",
        },
        PlotTextKey.QUADRANT_SIGNAL_EFF_YLABEL: {
            "en": "Signal Efficiency (TPR / $P_d$)",
            "pt": "Eficiência de Sinal (TPR / $P_d$)",
        },
        PlotTextKey.QUADRANT_SIGNAL_EFF_VS_TITLE: {
            "en": "Signal Class (Photons) Efficiency vs {var_label}",
            "pt": "Eficiência da Classe Sinal (Fótons) vs {var_label}",
        },
        PlotTextKey.QUADRANT_BG_EFF_YLABEL: {
            "en": r"Background Efficiency (TNR / $\mathrm{Eff}_{\mathrm{bg}}$)",
            "pt": r"Eficiência de Fundo (TNR / $\mathrm{Eff}_{\mathrm{bg}}$)",
        },
        PlotTextKey.QUADRANT_BG_EFF_VS_TITLE: {
            "en": "Background Class (Fakes) Efficiency vs {var_label}",
            "pt": "Eficiência da Classe Fundo (Falsos) vs {var_label}",
        },
        PlotTextKey.QUADRANT_EFF_SUPTITLE: {
            "en": "Classification Efficiency vs {var_label}\n{region_str} ({s1} vs {s2})",
            "pt": "Eficiência de Classificação vs {var_label}\n{region_str} ({s1} vs {s2})",
        },
        PlotTextKey.QUADRANT_SUMMARY_YLABEL: {
            "en": "Proportion of Holdout Events",
            "pt": "Proporção de Eventos de Holdout",
        },
        PlotTextKey.QUADRANT_SUMMARY_TITLE: {
            "en": "Regional Quadrant Distribution Summary\n{s1} vs {s2}",
            "pt": "Resumo da Distribuição Regional por Quadrantes\n{s1} vs {s2}",
        },
        PlotTextKey.QUADRANT_CLASS_EFF_SIG_TITLE: {
            "en": "Signal Class (Photons) Efficiency",
            "pt": "Eficiência da Classe Sinal (Fótons)",
        },
        PlotTextKey.QUADRANT_CLASS_EFF_BG_TITLE: {
            "en": "Background Class (Fakes) Efficiency",
            "pt": "Eficiência da Classe Fundo (Falsos)",
        },
        PlotTextKey.QUADRANT_CLASS_EFF_SUPTITLE: {
            "en": "Regional Classification Efficiencies by Class\n{s1} vs {s2}",
            "pt": "Eficiências Regionais de Classificação por Classe\n{s1} vs {s2}",
        },
        PlotTextKey.FEATURE_DIST_TRUE_SIGNAL: {
            "en": "True Signal",
            "pt": "Sinal Verdadeiro",
        },
        PlotTextKey.FEATURE_DIST_TRUE_BACKGROUND: {
            "en": "True Background",
            "pt": "Fundo Verdadeiro",
        },
        PlotTextKey.FEATURE_DIST_PREFIX_SIGNAL: {
            "en": "Signal",
            "pt": "Sinal",
        },
        PlotTextKey.FEATURE_DIST_PREFIX_BACKGROUND: {
            "en": "Jet Leak in Signal",
            "pt": "Vazamento de Jatos no Sinal",
        },
        PlotTextKey.FEATURE_DIST_SUBPLOT_TITLE: {
            "en": "{prefix}: {display_title}{title_suffix}",
            "pt": "{prefix}: {display_title}{title_suffix}",
        },
        PlotTextKey.FEATURE_DIST_EVENTS_YLABEL: {
            "en": "{prefix} Events / Bin",
            "pt": "Eventos de {prefix} / Bin",
        },
        PlotTextKey.FEATURE_DIST_RATIO_YLABEL: {
            "en": r"$N_{\mathrm{sel}} / N_{\mathrm{true}}$",
            "pt": r"$N_{\mathrm{sel}} / N_{\mathrm{true}}$",
        },
        PlotTextKey.FEATURE_DIST_OVERVIEW_HEADLINE_SIGNAL: {
            "en": "Signal (Photons)",
            "pt": "Sinal (Fótons)",
        },
        PlotTextKey.FEATURE_DIST_OVERVIEW_HEADLINE_BACKGROUND: {
            "en": "Jet Leakage in Signal",
            "pt": "Vazamento de Jatos no Sinal",
        },
        PlotTextKey.FEATURE_DIST_OVERVIEW_SUPTITLE: {
            "en": "Holdout {headline} Selection{suffix}",
            "pt": "Seleção de {headline} (Holdout){suffix}",
        },
        PlotTextKey.FEATURE_DIST_REGION_SUFFIX: {
            "en": " - Region {region_key}",
            "pt": " - Região {region_key}",
        },
        PlotTextKey.PILEUP_SIGNAL_EFFICIENCY: {
            "en": "Signal Efficiency ($P_D$)",
            "pt": "Eficiência de Sinal ($P_D$)",
        },
        PlotTextKey.PILEUP_BACKGROUND_FAKE_RATE: {
            "en": "Background Fake Rate ($P_F$)",
            "pt": "Taxa de Falsos Alarmes de Fundo ($P_F$)",
        },
        PlotTextKey.PILEUP_VS_MU_TITLE: {
            "en": "{metric_name} vs. $\\langle\\mu\\rangle$",
            "pt": "{metric_name} vs. $\\langle\\mu\\rangle$",
        },
        PlotTextKey.PILEUP_RATIO_TO: {
            "en": "Ratio to\n{ref_label}",
            "pt": "Razão para\n{ref_label}",
        },
        PlotTextKey.PILEUP_AVGMU_XLABEL: {
            "en": r"Average Interactions Per Crossing $\langle\mu\rangle$",
            "pt": r"Média de Interações por Cruzamento $\langle\mu\rangle$",
        },
        PlotTextKey.PILEUP_MEAN_OBSERVABLE_YLABEL: {
            "en": "Mean {observable_label}",
            "pt": "Média de {observable_label}",
        },
        PlotTextKey.PILEUP_EVENTS_PER_BIN: {
            "en": "Events / Bin",
            "pt": "Eventos / Bin",
        },
        PlotTextKey.PILEUP_MEAN_NPV_LABEL: {
            "en": r"Mean $N_{\text{PV}} \pm \text{SEM}$",
            "pt": r"Média $N_{\text{PV}} \pm \text{SEM}$",
        },
        PlotTextKey.PILEUP_FIT_NPV_LABEL: {
            "en": "Fit: $N_{{\\text{{PV}}}} = {slope:.2f}\\langle\\mu\\rangle + {intercept:.2f}$\n($R^2 = {r_sq:.3f}$)",
            "pt": "Ajuste: $N_{{\\text{{PV}}}} = {slope:.2f}\\langle\\mu\\rangle + {intercept:.2f}$\n($R^2 = {r_sq:.3f}$)",
        },
        PlotTextKey.PILEUP_VERTEX_TITLE: {
            "en": r"Primary Vertices $N_{\text{PV}}$ vs. $\langle\mu\rangle$",
            "pt": r"Vértices Primários $N_{\text{PV}}$ vs. $\langle\mu\rangle$",
        },
        PlotTextKey.PILEUP_NPV_YLABEL: {
            "en": r"Number of Good Primary Vertices ($N_{\text{PV}}$)",
            "pt": r"Número de Bons Vértices Primários ($N_{\text{PV}}$)",
        },
        PlotTextKey.PILEUP_SIGNAL_EVENTS_PER_BIN: {
            "en": "Signal Events / Bin",
            "pt": "Eventos de Sinal / Bin",
        },
        PlotTextKey.PILEUP_FIT_THRESHOLD_LABEL: {
            "en": "{name} Fit: $T(\\langle\\mu\\rangle) = {slope:.3f}\\langle\\mu\\rangle {sign} {abs_intercept:.3f}$ ($R^2 = {r2:.2f}$)",
            "pt": "Ajuste {name}: $T(\\langle\\mu\\rangle) = {slope:.3f}\\langle\\mu\\rangle {sign} {abs_intercept:.3f}$ ($R^2 = {r2:.2f}$)",
        },
        PlotTextKey.PILEUP_THRESHOLD_CALIB_TITLE: {
            "en": r"Discriminant $y_{\text{NN}}$ vs. $\langle\mu\rangle$ Threshold Calibration",
            "pt": r"Calibração de Limiar do Discriminante $y_{\text{NN}}$ vs. $\langle\mu\rangle$",
        },
        PlotTextKey.PILEUP_DISCRIMINANT_YLABEL: {
            "en": r"Neural Discriminant $y_{\text{NN}}$ (Logit)",
            "pt": r"Discriminante Neural $y_{\text{NN}}$ (Logit)",
        },
        PlotTextKey.EFFICIENCY_DEFAULT_FEATURE: {
            "en": "Feature",
            "pt": "Característica",
        },
        PlotTextKey.EFFICIENCY_DEFAULT_LABEL: {
            "en": "Efficiency (Signal)",
            "pt": "Eficiência (Sinal)",
        },
        PlotTextKey.EFFICIENCY_YLABEL: {
            "en": "Detection Efficiency (TPR)",
            "pt": "Eficiência de Detecção (TPR)",
        },
        PlotTextKey.EFFICIENCY_TITLE_VS: {
            "en": "Detection Efficiency vs {feature_name}",
            "pt": "Eficiência de Detecção vs {feature_name}",
        },
        PlotTextKey.EFFICIENCY_ET_STANDARD_FEATURE: {
            "en": "$E_T$ [GeV] (< {et_cutoff_gev} GeV)",
            "pt": "$E_T$ [GeV] (< {et_cutoff_gev} GeV)",
        },
        PlotTextKey.EFFICIENCY_ET_PROPOSED_FEATURE: {
            "en": "$E_T$ [GeV] (< {et_cutoff_gev} GeV) (Proposed Cut)",
            "pt": "$E_T$ [GeV] (< {et_cutoff_gev} GeV) (Corte Proposto)",
        },
        PlotTextKey.EFFICIENCY_ETA_STANDARD_FEATURE: {
            "en": "$\\eta$ (for $E_T$ < {et_cutoff_gev} GeV)",
            "pt": "$\\eta$ (para $E_T$ < {et_cutoff_gev} GeV)",
        },
        PlotTextKey.EFFICIENCY_ETA_PROPOSED_FEATURE: {
            "en": "$\\eta$ (for $E_T$ < {et_cutoff_gev} GeV) (Proposed Cut)",
            "pt": "$\\eta$ (para $E_T$ < {et_cutoff_gev} GeV) (Corte Proposto)",
        },
        PlotTextKey.EFFICIENCY_LABEL_STANDARD_CUT: {
            "en": "Efficiency (Threshold 0.5)",
            "pt": "Eficiência (Limiar 0.5)",
        },
        PlotTextKey.EFFICIENCY_LABEL_PROPOSED_CUT: {
            "en": "Efficiency (Proposed Cut)",
            "pt": "Eficiência (Corte Proposto)",
        },
        PlotTextKey.PEAK_PROFILE_TITLE: {
            "en": "Average Energy Profile with Top-{top_k} Peak Rings Highlighted ({n_rings} Rings) - {scope}",
            "pt": "Perfil Médio de Energia com Top-{top_k} Anéis de Pico Destacados ({n_rings} Anéis) - {scope}",
        },
        PlotTextKey.PEAK_INTRA_BAR_TITLE: {
            "en": "Top-{top_k} Rings Energy Share Within Each Subdetector (Intra-Detector %)",
            "pt": "Fração de Energia dos Top-{top_k} Anéis Dentro de Cada Subdetector (% Intra-Detector)",
        },
        PlotTextKey.PEAK_GLOBAL_BAR_TITLE: {
            "en": "Top-{top_k} Rings Energy Share Across Full Calorimeter (Global %)",
            "pt": "Fração de Energia dos Top-{top_k} Anéis no Calorímetro Total (% Global)",
        },
        PlotTextKey.PEAK_BAR_SUPTITLE: {
            "en": "Subdetector Peak Ring Energy Concentration ({n_rings} Rings) - {scope}",
            "pt": "Concentração de Energia nos Anéis de Pico por Subdetector ({n_rings} Anéis) - {scope}",
        },
        PlotTextKey.PEAK_SUBDETECTOR_XLABEL: {
            "en": "Calorimeter Subdetector",
            "pt": "Subdetector do Calorímetro",
        },
        PlotTextKey.PEAK_INTRA_YLABEL: {
            "en": "Intra-Detector Energy Share (%)",
            "pt": "Fração de Energia Intra-Detector (%)",
        },
        PlotTextKey.PEAK_GLOBAL_YLABEL: {
            "en": "Total Calorimeter Energy Share (%)",
            "pt": "Fração da Energia Total do Calorímetro (%)",
        },
        PlotTextKey.PEAK_HEATMAP_TITLE: {
            "en": "Regional Top-{top_k} Peak Energy Share ({metric_label}) - {class_label}",
            "pt": "Fração Regional de Energia nos Top-{top_k} Anéis ({metric_label}) - {class_label}",
        },
        PlotTextKey.PEAK_REGION_XLABEL: {
            "en": "Calorimeter Subdetector",
            "pt": "Subdetector do Calorímetro",
        },
        PlotTextKey.PEAK_REGION_YLABEL: {
            "en": "Kinematic Region ($E_T, \\eta$)",
            "pt": "Região Cinemática ($E_T, \\eta$)",
        },
        PlotTextKey.PEAK_HIGHLIGHT_LABEL: {
            "en": "Top-{top_k} Peak Rings",
            "pt": "Top-{top_k} Anéis de Pico",
        },
        PlotTextKey.PEAK_TOTAL_SUBDET_LABEL: {
            "en": "Full Subdetector",
            "pt": "Subdetector Completo",
        },
    }

    @classmethod
    def validate_language(cls, language: str) -> SupportedLanguage:
        """Validates that a language code is supported for single-language rendering.

        Args:
            language: Language identifier string ('en' or 'pt').

        Returns:
            Validated language identifier.

        Raises:
            ValueError: If language is not 'en' or 'pt'.
        """
        if language not in cls.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Supported languages: {cls.SUPPORTED_LANGUAGES}."
            )
        return language  # type: ignore[return-value]

    @classmethod
    def translate(cls, key: Union[PlotTextKey, str], language: str = "en", **kwargs: Any) -> str:
        """Looks up and formats a localized string template for a given translation key.

        Args:
            key: Key identifying the text entry in TRANSLATIONS.
            language: Target language code ('en' or 'pt').
            **kwargs: Optional format placeholders interpolated into the template.

        Returns:
            Localized and formatted text string.

        Raises:
            ValueError: If language is unsupported.
            KeyError: If key is not present in TRANSLATIONS.
        """
        validated_lang = cls.validate_language(language)
        if key not in cls.TRANSLATIONS:
            raise KeyError(f"Unknown translation key: '{key}'.")
        template = cls.TRANSLATIONS[key][validated_lang]
        if kwargs:
            return template.format(**kwargs)
        return template


class BilingualPlotOrchestrator:
    """Coordinates plotting execution across single ('en', 'pt') or bilingual ('both') modes.

    Attributes:
        language: Configured language option ('en', 'pt', or 'both').
        languages: Ordered list of target single-language codes to execute.
        is_multi_language: True when executing for multiple languages ('both').
    """

    SUPPORTED_OPTIONS: ClassVar[Tuple[str, ...]] = ("en", "pt", "both")

    def __init__(self, language: str = "en") -> None:
        """Initializes BilingualPlotOrchestrator with a validated language option.

        Args:
            language: Language option ('en', 'pt', or 'both').
        """
        self.language: str = language
        self.languages: List[str] = self.resolve_languages(language)
        self.is_multi_language: bool = len(self.languages) > 1

    @staticmethod
    def resolve_languages(language: str) -> List[str]:
        """Resolves a language option string into concrete language codes.

        Args:
            language: Language option ('en', 'pt', or 'both').

        Returns:
            List of concrete language codes (['en'], ['pt'], or ['en', 'pt']).

        Raises:
            ValueError: If language is not 'en', 'pt', or 'both'.
        """
        if language == "en":
            return ["en"]
        if language == "pt":
            return ["pt"]
        if language == "both":
            return ["en", "pt"]
        raise ValueError(
            f"Unsupported language option '{language}'. Expected one of: 'en', 'pt', 'both'."
        )

    @staticmethod
    def resolve_output_dir(
        base_output_dir: Union[str, Path],
        lang: str,
        is_multi_language: bool,
    ) -> Path:
        """Resolves the target output directory for a single language execution.

        Args:
            base_output_dir: Root directory configured for plot export.
            lang: Concrete language code ('en' or 'pt').
            is_multi_language: True if multiple languages are being generated.

        Returns:
            Subdirectory `<base_output_dir>/<lang>` when multi-language, else `<base_output_dir>`.
        """
        PlotTranslator.validate_language(lang)
        base_path = Path(base_output_dir)
        if is_multi_language:
            return base_path / lang
        return base_path

    def orchestrate(
        self,
        base_output_dir: Union[str, Path],
        task_fn: Callable[[str, Path], Any],
    ) -> Dict[str, Any]:
        """Executes a plotting callback across all resolved languages.

        Args:
            base_output_dir: Root directory where plots should be saved.
            task_fn: Callable accepting `(lang, resolved_output_dir)` and returning any result.

        Returns:
            Dictionary mapping each executed language code to its callback return value.
        """
        results: Dict[str, Any] = {}
        for lang in self.languages:
            resolved_dir = self.resolve_output_dir(
                base_output_dir=base_output_dir,
                lang=lang,
                is_multi_language=self.is_multi_language,
            )
            results[lang] = task_fn(lang, resolved_dir)
        return results
