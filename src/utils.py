from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import os
from typing import Union, Callable, Tuple 
from typing import Union, Callable, Tuple 
import logging
log = logging.getLogger()

def create_folder(new_folder_name: str , base_path: str = "results") -> str:
    folder_path = Path(base_path) / new_folder_name 
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Folder '{folder_path}' created or already exists (using pathlib).")
    except OSError as error:
        log.error(f"Error creating directory '{folder_path}': {error}")
    return str(folder_path)


def norm1(data: np.ndarray) -> np.ndarray:
      norms = np.abs( data.sum(axis=1) )
      norms[norms==0] = 1
      return data / norms[:, None]


# def plot_profile_mean_energy_rings(signal_data:  np.ndarray, bg_data:  np.ndarray,
#                                    folder_path: str, iet: int, ieta: int,
#                                   percentage: float=1.) -> None:
#     signal = norm1(signal_data)
#     background = norm1(bg_data)

#     x_axis = np.arange(len(signal[0]))
#     number_of_rings = len(signal[0,:])
#     mean_ringsSignal = np.mean(signal, axis=0)
#     std_ringsSignal = np.std(signal, axis=0)

#     mean_ringBKG = np.mean(background, axis=0)
#     std_ringBKG = np.std(background, axis=0)
    
#     yAxis_max = np.max(mean_ringBKG) if np.max(mean_ringBKG) > np.max(mean_ringsSignal) else np.max(mean_ringsSignal)
#     # Subdetectores e cores
#     subdet_names = ['PreSampler', 'EM1', 'EM2', 'EM3', 'TileCal']
#     indexes = np.array(get_rings_index(percentage))
#     subdet_x = [0, 
#                 np.where(indexes == 8)[0][0],
#                 np.where(indexes == 72)[0][0],
#                 np.where(indexes == 80)[0][0],
#                 np.where(indexes == 88)[0][0]]
#     subdet_colors = ['#1b9e77', '#d95f09', '#7570b3', '#e7298a', '#66a61e']

#     plt.figure(figsize=(10, 5), clear=True, num=1)
#     # --- Curva Fóton ---
#     plt.errorbar(x_axis, mean_ringsSignal, std_ringsSignal,
#                  marker='o', mfc='navy', mec='navy', ms=3,
#                  mew=0.5, elinewidth=0.8, capsize=2,
#                  ecolor='navy', color='navy',
#                  label='Fóton')
#     # --- Curva Jatos Hadrônicos ---
#     plt.errorbar(x_axis, mean_ringBKG, std_ringBKG,
#                  marker='s', mfc='darkorange', mec='darkorange', ms=3,
#                  mew=0.5, elinewidth=0.8, capsize=2,
#                  ecolor='darkorange', color='darkorange',
#                  label='Jatos Hadrônicos')
#     # --- Linhas verticais e rótulos mais baixos (y=0.66) ---
#     for x, name, color in zip(subdet_x, subdet_names, subdet_colors):
#         plt.axvline(x=x, color=color, linestyle='--', linewidth=1)
#         plt.text(x+1.2, yAxis_max * 1.2, name, rotation=90,
#                  va='bottom', ha='center', fontsize=9, color=color)
#     # --- Estilo dos eixos ---
#     plt.xlabel('Anéis', fontsize=13)
#     plt.ylabel('Energia Normalizada', fontsize=13)
#     plt.xticks(ticks=np.linspace(0, number_of_rings, 10, dtype=int),
#                labels=[str(i+1) for i in np.linspace(0, number_of_rings, 10, dtype=int)],
#                fontsize=11)
#     plt.yticks(fontsize=11)
#     # plt.xlim(-1, number_of_rings)
#     plt.ylim(-0.05, yAxis_max * 1.5)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     # --- Legenda sinal/fundo ---
#     plt.legend(fontsize=10, loc='upper right')
#     plt.title('Perfil Médio de Energia nos Anéis - NeuralRinger', fontsize=14)
#     # --- Salvar com bounding box que inclui textos externos ---
#     plt.tight_layout()
#     try:
#         plt.savefig(os.path.join(folder_path,
#                                  f"et{iet}.eta{ieta}.RingsMeanProfiles_Ylim_OK_Rotulos_OK_NeuralRinger.pdf"),
#                     format='pdf',
#                     dpi=300,
#                     transparent=True, 
#                     bbox_inches='tight')
#         # plt.show()
#         # https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots
#         # plt.close()
#     except Exception as e:
#         # plt.close()
#         raise
        
#     df = pd.DataFrame({
#         'signal_data': signal_data,
#         'bg_data': bg_data,
#     })
#     df.to_pickle(os.path.join(folder_path, 'plot_profile_mean_energy_rings.pkl'))


def plot_boxplot_SP(data_type: str,
                    all_training_results:  dict[str, Union[str, int, object]],
                    folder_path: str, iet: int, ieta: int,
                   should_plot = False) -> None:
    if should_plot:
        if not all_training_results:
            print("Erro: A lista 'all_training_results' está vazia. Por favor, execute o treinamento primeiro.")
        else:
            if data_type == "best_sp_value":
                title = 'Distribuição do Índice SP por Fold (Agregado pelas Inicializações)'
                y_label = 'Melhor Índice SP'
            if data_type == "best_fa_value":
                title = 'Distribuição do FA por Fold (Agregado pelas Inicializações)'
                y_label = 'Fake Rate'
            if data_type == "best_pd_value":
                title = 'Distribuição do PD por Fold (Agregado pelas Inicializações)'
                y_label = 'Efficiency'
            # --- 1. Converter a lista de dicionários em um DataFrame Pandas ---
            df_results = pd.DataFrame(all_training_results)
        
            # Filtrar qualquer entrada onde 'best_sp_value' seja None (se houver)
            df_results = df_results.dropna(subset=[data_type])
            if df_results.empty:
                print("Nenhum dado válido encontrado para plotar boxplots.")
            else:
                # --- 2. Plotar os Boxplots do Índice SP por Fold ---
                plt.figure(figsize=(12, 7), clear=True, num=1)
                # 'x='fold'' agrupará os boxplots por fold.
                # 'y='best_sp_value'' será a métrica de cada boxplot.
                # 'data=df_results' especifica o DataFrame.
                sns.boxplot(x='fold', y=data_type, data=df_results, palette='plasma',
                            hue='fold', legend=False)
                # O swarmplot adiciona os pontos individuais, mostrando a densidade dos dados.
                sns.swarmplot(x='fold', y=data_type, data=df_results, 
                              palette='dark:black', alpha=0.7, size=3, 
                              hue='fold', legend=False)
        
                plt.title(title)
                plt.xlabel('Número do Fold')
                plt.ylabel(y_label)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                # Ajusta os rótulos do eixo X para serem mais legíveis, começando do Fold 1
                plt.xticks(ticks=range(df_results['fold'].max() + 1), 
                           labels=[f'Fold {f+1}' for f in range(df_results['fold'].max() + 1)])
                plt.tight_layout()
                fold_file_template = "iet{iet}.ieta{ieta}_{data_type}.pdf".format(
                    ieta = ieta, 
                    iet = iet,
                    data_type = data_type)
                plt.savefig(os.path.join(folder_path, fold_file_template))
                # plt.show()
                # plt.close()
    else:
        df_results = pd.DataFrame(all_training_results)
        df_results.to_pickle(
            os.path.join(folder_path, f'iet{iet}.ieta{ieta}.{data_type}_plot_boxplot_SP.pkl'))

# def get_best_sp_model(training_results_list: list[dict],
#                       model_builder_tag: str,
#                       input_dimensions: int) ->  Tuple[tf.keras.Sequential, 
#                                                 dict[str, Union[str, int, object]]]:
#     """
#     Identifica e retorna o modelo Keras com o maior índice SP de uma lista de resultados de treinamento.

#     Args:
#         training_results_list (list): Uma lista de dicionários, onde cada dicionário
#                                       contém 'fold', 'repeat', 'best_sp_value',
#                                       'best_weights' e 'keras_history'.
#         model_builder_fn (function): Uma função que reconstrói a arquitetura do modelo
#                                      (ex: build_model(input_dim)).
#         input_dimensions (int): A dimensão de entrada necessária para construir o modelo.

#     Returns:
#         tf.keras.Model: O modelo Keras com os pesos correspondentes ao maior SP.
#         dict: As informações completas da rodada que produziu o melhor modelo.
#               Retorna (None, None) se a lista estiver vazia ou nenhum modelo foi salvo.
#     """
#     if not training_results_list:
#         print("A lista de resultados de treinamento está vazia.")
#         return None, None

#     highest_sp_model_info = None
#     max_sp_index_global = -float('inf')

#     for run_info in training_results_list:
#         current_sp = run_info.get('best_sp_value', -float('inf'))

#         # Ignora entradas onde best_weights não foi salvo (ex: devido a um SP inicial de 0)
#         if run_info.get('best_weights') is None:
#             continue

#         if current_sp > max_sp_index_global:
#             max_sp_index_global = current_sp
#             highest_sp_model_info = run_info

#     if highest_sp_model_info:
#         # print(f"\n--- Modelo com o mais alto índice SP encontrado ---")
#         # print(f"  Fold: {highest_sp_model_info['fold']+1}")
#         # print(f"  Inicialização: {highest_sp_model_info['repeat']+1}")
#         # print(f"  Melhor Índice SP: {highest_sp_model_info['best_sp_value']:.4f}")

#         # Recriar a arquitetura do modelo
#         best_overall_sp_model = build_model(input_dimensions, model_builder_tag)

#         # Carregar os melhores pesos
#         best_overall_sp_model.set_weights(highest_sp_model_info['best_weights'])

#         # print("Modelo recuperado com sucesso e pronto para uso!")
#         return best_overall_sp_model, highest_sp_model_info
#     else:
#         print("Nenhum modelo com pesos salvos foi encontrado na lista fornecida.")
#         return None, None


def plot_model_metrics(best_model_details: dict[str, Union[str, int, object]]) -> None:
    if best_model_details is None:
        print("Erro: Detalhes do melhor modelo não encontrados. Por favor, execute 'get_best_sp_model' primeiro.")
    else:
        # --- 1. Extrair os Dados do Histórico ---
        history_data = best_model_details
    
        epochs = range(1, len(history_data['train_loss']) + 1)
    
        # Métricas comuns
        train_loss = history_data['train_loss']
        val_loss = history_data['val_loss']
        train_accuracy = history_data['train_acc']
        val_accuracy = history_data['val_acc']
    
        # Valores de SP, FPR e TPR do callback SP
        callbackMetrics = history_data["callbackMetrics"][0]
        val_sp = callbackMetrics.get('max_sp_val') # .get() para evitar KeyError se a chave não existir
        val_fa = callbackMetrics.get('max_sp_fa_val') # FPR
        val_pd = callbackMetrics.get('max_sp_pd_val') # TPR

        # Verifica se as métricas do SP estão presentes
        if val_sp is None or val_fa is None or val_pd is None:
            print("Aviso: Algumas métricas (SP, FPR, TPR) não encontradas no histórico. Verifique a implementação do seu callback 'sp'.")
            # Podemos definir como arrays vazios ou None para que os plots condicionais funcionem
            val_sp, val_fa, val_pd = None, None, None
    
        # --- 2. Plotar as Curvas de Aprendizagem (Loss, Accuracy, e SP/FPR/TPR) ---
        plt.figure(figsize=(18, 6), clear=True, num=1) # Aumenta a figura para 3 subplots
    
        # Plot Loss
        plt.subplot(1, 3, 1) # 1 linha, 3 colunas, 1º plot
        plt.plot(epochs, train_loss, 'o-', label='Perda no Treinamento')
        plt.plot(epochs, val_loss, 'o-', label='Perda na Validação')
        plt.title('Curvas de Perda (Loss)')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.grid(True)
        plt.legend()
    
        # Plot Accuracy
        plt.subplot(1, 3, 2) # 1 linha, 3 colunas, 2º plot
        plt.plot(epochs, train_accuracy, 'o-', label='Acurácia no Treinamento')
        plt.plot(epochs, val_accuracy, 'o-', label='Acurácia na Validação')
        plt.title('Curvas de Acurácia')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.grid(True)
        plt.legend()
    
        # Plot SP, TPR e FPR (se disponíveis)
        plt.subplot(1, 3, 3) # 1 linha, 3 colunas, 3º plot
        if val_sp is not None:
            plt.plot(epochs, val_sp, 'o-', label='Índice SP', color='purple')
        if val_pd is not None:
            plt.plot(epochs, val_pd, 'o-', label='TPR (Verdadeiros Positivos)', color='darkgreen')
        if val_fa is not None:
            plt.plot(epochs, val_fa, 'o-', label='FPR (Falsos Positivos)', color='darkred')
    
        if val_sp is not None:
            # Linha vertical para o melhor SP
            best_sp_epoch_idx = np.argmax(val_sp)
            plt.axvline(x=epochs[best_sp_epoch_idx], color='blue', linestyle='--', label=f'Melhor SP (Época {epochs[best_sp_epoch_idx]})')
    
        plt.title('Métricas de Desempenho (SP, TPR, FPR)')
        plt.xlabel('Época')
        plt.ylabel('Valor da Métrica')
        plt.grid(True)
        plt.legend()
    
        plt.tight_layout()
        # model_metrics_file_template = "iet{iet}.ieta{ieta}_modelMetrics.pdf".format(
        #     ieta = ieta, 
        #     iet=iet)
        # plt.savefig(os.path.join(folder_path, model_metrics_file_template))
        plt.show()
        # plt.close()
    
        # print("\nCurvas de aprendizagem (Loss, Accuracy, SP, TPR, FPR) geradas com sucesso para o melhor modelo SP!")


# def plot_model_acc(best_overall_sp_model: tf.keras.Sequential, 
#                    test_set: Tuple[np.ndarray,np.ndarray],
#                   folder_path=None, ns = "", iet="", ieta="") -> float:
#     X_test_final, y_test_final = test_set
#     if best_overall_sp_model is None:
#         print("Erro: O modelo com o melhor SP não foi encontrado. Certifique-se de que 'get_best_sp_model' foi executada corretamente.")
#     else:
#         # --- 2. Previsões da Rede ---
#         print("\nFazendo previsões no conjunto de teste com o modelo de melhor SP...")
#         y_pred_proba = best_overall_sp_model.predict(X_test_final).ravel()
    
#         # --- 3. Curva ROC ---
#         print("Gerando Curva ROC...")
#         fpr, tpr, thresholds = roc_curve(y_test_final, y_pred_proba)
#         auc_score = roc_auc_score(y_test_final, y_pred_proba)
    
#         plt.figure(figsize=(8, 6), clear=True, num=1)
#         plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {auc_score:.4f})')
#         plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('Taxa de Falsos Positivos (FPR)')
#         plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
#         plt.title('Curva ROC do Modelo'+ \
#              f"\n({iet},{ieta})")
#         plt.legend(loc="lower right")
#         plt.grid(True)
#         roc_file_template = "ns{ns}.iet{iet}.ieta{ieta}_ROC.pdf".format(
#             ns = ns,
#             ieta = ieta, 
#             iet=iet)
#         plt.savefig(os.path.join(folder_path, roc_file_template))
#         # plt.show()
#         # plt.close()
#         # --- 4. Histograma de Saída da Rede (Sinal vs. Background) ---
#         print("Gerando Histograma de Saída da Rede...")
    
#         # Separar as previsões para sinal e background
#         # Assumindo que 1 é sinal e 0 é background em y_test_final
#         predictions_signal = y_pred_proba[y_test_final == 1]
#         predictions_background = y_pred_proba[y_test_final == 0]
    
#         plt.figure(figsize=(10, 6), clear=True, num=1)
#         plt.hist(predictions_background, bins=50, alpha=0.7, label='Background', color='red', density=True)
#         plt.hist(predictions_signal, bins=50, alpha=0.7, label='Sinal', color='green', density=True)
#         plt.xlabel('Saída da Rede (Probabilidade)')
#         plt.ylabel('Densidade')
#         plt.yscale('log')
#         plt.title('Histograma de Saída da Rede para Sinal e Background'+ \
#              f"\n({iet},{ieta})")
#         plt.legend(loc='upper right')
#         plt.grid(True)
#         model_output_file_template = "ns{ns}.iet{iet}.ieta{ieta}_modelOutput.pdf".format(
#             ns = ns,
#             ieta = ieta, 
#             iet=iet)
#         plt.savefig(os.path.join(folder_path, model_output_file_template))
#         # plt.show()
#         # plt.close()
    
#         print("\nVisualizações geradas com sucesso!")
#         return auc_score


# def compute_saliency_map(x: np.ndarray, model: tf.keras.Sequential) -> np.ndarray:
#     x_tensor = tf.convert_to_tensor(x[np.newaxis, :], dtype=tf.float32)
#     with tf.GradientTape() as tape:
#         tape.watch(x_tensor)
#         pred = model(x_tensor)
#     grad = tape.gradient(pred, x_tensor)
#     saliency = np.abs(grad.numpy()[0])
#     # Normalização Min-Max por amostra
#     smin, smax = saliency.min(), saliency.max()
#     saliency_norm = (saliency - smin) / (smax - smin + 1e-8)
#     return saliency_norm


# def compute_mean_std_saliency(X_subset: np.ndarray, 
#                               model: tf.keras.Sequential) -> Tuple[float, float]:
#     saliency_list = [compute_saliency_map(x, model) for x in X_subset]
#     saliency_array = np.stack(saliency_list)
#     return np.mean(saliency_array, axis=0), np.std(saliency_array, axis=0)


# def plot_saliency_comparison_normalized(model: tf.keras.Sequential, 
#                                         X_test: np.ndarray, y_test: np.ndarray, 
#                                         n_samples=100,
#                                         folder_path=None,
#                                        ns = "", iet="", ieta="") -> None:
#     # --- Seleciona n_samples de cada classe ---
#     idx_sinal = np.where(y_test == 1)[0]
#     idx_backg = np.where(y_test == 0)[0]
#     np.random.shuffle(idx_sinal)
#     np.random.shuffle(idx_backg)

#     idx_sinal = idx_sinal[:n_samples]
#     idx_backg = idx_backg[:n_samples]
#     X_sinal = X_test[idx_sinal]
#     X_backg = X_test[idx_backg]
#     # --- Calcula perfis de saliency normalizados ---
#     mean_sinal, std_sinal = compute_mean_std_saliency(X_sinal, model)
#     mean_backg, std_backg = compute_mean_std_saliency(X_backg, model)
#     # --- Plot ---
#     plt.figure(figsize=(12, 6), clear=True, num=1)
#     x_range = np.arange(len(mean_sinal))

#     plt.plot(mean_sinal, label='Sinal (Média)', color='blue')
#     plt.fill_between(x_range, 
#                      mean_sinal - std_sinal,
#                      mean_sinal + std_sinal,
#                      color='blue', alpha=0.3, label='Sinal ±1σ')
#     plt.plot(mean_backg, label='Background (Média)', color='red')
#     plt.fill_between(x_range, 
#                      mean_backg - std_backg, 
#                      mean_backg + std_backg,
#                      color='red', alpha=0.3, label='Background ±1σ')
#     plt.title(f'Comparação do Perfil Médio de Saliency Normalizado\n({n_samples} amostras por classe)' + \
#              f"\n({iet},{ieta})")
#     plt.xlabel('Posição da Entrada - Indice de Anéis')
#     plt.ylabel('Importância Normalizada (Saliency)')
#     plt.legend()
#     plt.grid(True, linestyle='--')
#     plt.tight_layout()
    
#     saliency_file_template = "ns{ns}.iet{iet}.ieta{ieta}_saliency_comparacao_normalizada.pdf".format(
#         ns = ns,
#         ieta = ieta, 
#         iet=iet)
#     plt.savefig(os.path.join(folder_path, saliency_file_template), dpi=300)
#     # plt.show()
#     # plt.close()











