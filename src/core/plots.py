import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Union
import torch
import os
from src.utils import create_folder, compute_mean_std_saliency
import logging
log = logging.getLogger()


def plot_profile_mean_energy_rings(data:  np.ndarray,
                                   target:  np.ndarray,
                                   ring_index: np.ndarray,
                                   folder_path: str,
                                   iet: int = 0, 
                                   ieta: int = 0) -> None:
    
    signal_data = data[np.where(target == 1)]
    bg_data = data[np.where(target == 0)]
    ring_index = np.array(ring_index)
    x_axis = np.arange(len(signal_data[0]))
    number_of_rings = len(signal_data[0,:])
    mean_ringsSignal = np.mean(signal_data, axis=0)
    std_ringsSignal = np.std(signal_data, axis=0)
    mean_ringBKG = np.mean(bg_data, axis=0)
    std_ringBKG = np.std(bg_data, axis=0)
    
    yAxis_max = np.max(mean_ringBKG) if np.max(mean_ringBKG) > np.max(mean_ringsSignal) else np.max(mean_ringsSignal)
    subdet_names = ['PreSampler', 'EM1', 'EM2', 'EM3', 'TileCal']
    subdet_x = [0, 
                np.where(ring_index == 8)[0][0],
                np.where(ring_index == 72)[0][0],
                np.where(ring_index == 80)[0][0],
                np.where(ring_index == 88)[0][0]]
    subdet_colors = ['#1b9e77', '#d95f09', '#7570b3', '#e7298a', '#66a61e']

    plt.figure(figsize=(10, 5), clear=True, num=1)
    plt.errorbar(x_axis, mean_ringsSignal, std_ringsSignal,
                 marker='o', mfc='navy', mec='navy', ms=3,
                 mew=0.5, elinewidth=0.8, capsize=2,
                 ecolor='navy', color='navy',
                 label='Photons')
    plt.errorbar(x_axis, mean_ringBKG, std_ringBKG,
                 marker='s', mfc='darkorange', mec='darkorange', ms=3,
                 mew=0.5, elinewidth=0.8, capsize=2,
                 ecolor='darkorange', color='darkorange',
                 label='Hadronic Jets')
    for x, name, color in zip(subdet_x, subdet_names, subdet_colors):
        plt.axvline(x=x, color=color, linestyle='--', linewidth=1)
        plt.text(x+1.2, yAxis_max * 1.2, name, rotation=90,
                 va='bottom', ha='center', fontsize=9, color=color)
    plt.xlabel('rings', fontsize=13)
    plt.ylabel('Normalized Energy', fontsize=13)
    plt.xticks(ticks=np.linspace(0, number_of_rings, 10, dtype=int),
               labels=[str(i+1) for i in np.linspace(0, number_of_rings, 10, dtype=int)],
               fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(-0.05, yAxis_max * 1.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.title('Average Energy Profile in the Rings - NeuralRinger', fontsize=14)
    plt.tight_layout()
    try:
        path = create_folder("RingsMeanProfiles", folder_path )
        plt.savefig(os.path.join(path, f"et{iet}.eta{ieta}.RingsMeanProfiles_NeuralRinger.pdf"),
                    format='pdf',
                    dpi=300,
                    transparent=True, 
                    bbox_inches='tight')
        plt.close()
    except Exception as e:
        plt.close()
        raise e

def plot_boxplot_SP(data_type: str,
                    all_training_results: dict[str, Union[str, int, object]],
                    folder_path: str, iet: int, ieta: int) -> None:
    if not all_training_results:
        log.error("The 'all_training_results' list is empty. Please run the training first.")
    else:
        if data_type == "best_sp_value":
            title = 'Distribution of SP Index by Fold (Aggregated by Initializations)'
            y_label = 'Best SP Index'
        if data_type == "best_fa_value":
            title = 'Distribution of FA by Fold (Aggregated by Initializations)'
            y_label = 'Fake Rate'
        if data_type == "best_pd_value":
            title = 'Distribution of PD by Fold (Aggregated by Initializations)'
            y_label = 'Efficiency'
        df_results = pd.DataFrame(all_training_results)
    
        df_results = df_results.dropna(subset=[data_type])
        if df_results.empty:
            log.warning("No valid data found to plot boxplots.")
        else:
            plt.figure(figsize=(12, 7), clear=True, num=1)
            sns.boxplot(x='fold', y=data_type, data=df_results, palette='plasma',
                        hue='fold', legend=False)
            sns.swarmplot(x='fold', y=data_type, data=df_results, 
                            palette='dark:black', alpha=0.7, size=3, 
                            hue='fold', legend=False)
            plt.title(title)
            plt.xlabel('Fold Number')
            plt.ylabel(y_label)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(ticks=range(df_results['fold'].max() + 1), 
                        labels=[f'Fold {f+1}' for f in range(df_results['fold'].max() + 1)])
            try:
                path = create_folder("boxplot_SP", folder_path )
                plt.savefig(os.path.join(path, f"et{iet}.eta{ieta}.boxplot_SP.pdf"),
                            format='pdf',
                            dpi=300,
                            transparent=True, 
                            bbox_inches='tight')
                plt.close()
            except Exception as e:
                plt.close()
                raise e
            
def plot_model_metrics(best_model_details: dict[str, Union[str, int, object]],
                       folder_path: str,
                       iet: int,
                       ieta: int) -> None:
    if best_model_details is None:
        log.error("Best model details not found. Please run 'get_best_sp_model' first.")
    else:
        history_data = best_model_details
        epochs = range(1, len(history_data['train_loss']) + 1)
    
        train_loss = history_data['train_loss']
        val_loss = history_data['val_loss']
        train_accuracy = history_data['train_acc']
        val_accuracy = history_data['val_acc']
    
        callbackMetrics = history_data["callbackMetrics"]
        val_sp = callbackMetrics.get('max_sp_val') 
        val_fa = callbackMetrics.get('max_sp_fa_val') 
        val_pd = callbackMetrics.get('max_sp_pd_val') 

        if val_sp is None or val_fa is None or val_pd is None:
            log.warning("Some metrics (SP, FPR, TPR) not found in history. Check your 'sp' callback implementation.")
            val_sp, val_fa, val_pd = None, None, None
    
        plt.figure(figsize=(18, 6), clear=True, num=1) # Increases figure size for 3 subplots
    
        plt.subplot(1, 3, 1) 
        plt.plot(epochs, train_loss, 'o-', label='Training Loss')
        plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
    
        plt.subplot(1, 3, 2) 
        plt.plot(epochs, train_accuracy, 'o-', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'o-', label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
    
        plt.subplot(1, 3, 3) 
        if val_sp is not None:
            plt.plot(epochs, val_sp, 'o-', label='SP metric', color='purple')
        if val_pd is not None:
            plt.plot(epochs, val_pd, 'o-', label='TPR (True Positives)', color='darkgreen')
        if val_fa is not None:
            plt.plot(epochs, val_fa, 'o-', label='FPR (False Positives)', color='darkred')
    
        if val_sp is not None:
            best_sp_epoch_idx = np.argmax(val_sp)
            plt.axvline(x=epochs[best_sp_epoch_idx], color='blue', linestyle='--', label=f'Best SP (Epoch {epochs[best_sp_epoch_idx]})')
    
        plt.title('Performance Metrics (SP, TPR, FPR)')
        plt.xlabel('Epoch')
        plt.ylabel('Metric value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    
        try:
            path = create_folder("model_metrics", folder_path )
            plt.savefig(os.path.join(path, f"iet{iet}.ieta{ieta}_model_metrics.pdf"),
                        format='pdf',
                        dpi=300,
                        transparent=True, 
                        bbox_inches='tight')
            plt.close()
        except Exception as e:
            raise e

def plot_model_acc(best_model_details,
                   folder_path: str, iet: int, ieta: int) -> None:
          log.info("Generating ROC Curve...")
          tpr, fpr  = best_model_details["pd"], best_model_details["fa"]
          auc_score = best_model_details["auc_score"]

          plt.figure(figsize=(8, 6), clear=True, num=1)
          plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {auc_score:.4f})')
          plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('False Positive Rate (FPR)')
          plt.ylabel('True Positive Rate (TPR)')
          plt.title('Model ROC Curve'+ \
               f"\n({iet},{ieta})")
          plt.legend(loc="lower right")
          plt.grid(True)
          log.info("Generating Network Output Histogram")
          try:
               path = create_folder("ROC", folder_path )
               plt.savefig(os.path.join(path, f"iet{iet}.ieta{ieta}_model_ROC.pdf"),
                         format='pdf',
                         dpi=300,
                         transparent=True, 
                         bbox_inches='tight')
               # https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots
               plt.close()
          except Exception as e:
               raise e

def plot_saliency_comparison_normalized(model: torch.nn.Module, 
                                        X_test: np.ndarray,
                                        y_test: np.ndarray,
                                        folder_path: str,
                                        iet: int, ieta: int,
                                        n_samples=100) -> None:
    idx_sinal = np.where(y_test == 1)[0]
    idx_backg = np.where(y_test == 0)[0]
    np.random.shuffle(idx_sinal)
    np.random.shuffle(idx_backg)

    idx_sinal = idx_sinal[:n_samples]
    idx_backg = idx_backg[:n_samples]
    X_sinal = X_test[idx_sinal]
    X_backg = X_test[idx_backg]
    mean_sinal, std_sinal = compute_mean_std_saliency(X_sinal, model)
    mean_backg, std_backg = compute_mean_std_saliency(X_backg, model)
    plt.figure(figsize=(12, 6), clear=True, num=1)
    x_range = np.arange(len(mean_sinal))

    plt.plot(mean_sinal, label='Signal (mean)', color='blue')
    plt.fill_between(x_range, 
                     mean_sinal - std_sinal,
                     mean_sinal + std_sinal,
                     color='blue', alpha=0.3, label='Signal ±1σ')
    plt.plot(mean_backg, label='Background (mean)', color='red')
    plt.fill_between(x_range, 
                     mean_backg - std_backg, 
                     mean_backg + std_backg,
                     color='red', alpha=0.3, label='Background ±1σ')
    plt.title(f'Comparison of Normalized Mean Saliency Profile\n({n_samples} samples per class)' + \
              f"\n({iet},{ieta})")
    plt.xlabel('Input Position - Ring Index')
    plt.ylabel('Normalized Importance (Saliency)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    try:
        path = create_folder("saliency", folder_path )
        plt.savefig(os.path.join(path, f"iet{iet}.ieta{ieta}_saliency_comparison_normalized.pdf"),
                    format='pdf',
                    dpi=300,
                    transparent=True, 
                    bbox_inches='tight')
        plt.close()
    except Exception as e:
        raise e









