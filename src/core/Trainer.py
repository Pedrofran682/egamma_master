import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from src.Architectures.SPCallbackPyTorch import SPCallbackPyTorch
from src.Architectures.Models import get_model
from datetime import datetime
import os
from src.utils import create_folder
from src.core.EgammaNpzDataset import EgammaNpzDataset
from src.utils import verify_results, get_results_file_name, get_et_eta, get_best_sp_model
from src.core.plots import plot_saliency_comparison_normalized
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import ParameterGrid
import logging
log = logging.getLogger()
from torch.utils.data import Dataset


class Trainer:
    def __init__(self,
                 n_splits=10,
                 folder_path=None, 
                 batch_size=256, 
                 epochs=50,
                 percentage=1, 
                 model_tag="V0",
                 et_range = np.arange(0, 2), 
                 eta_range = np.arange(0, 1),
                 debug=False,
                 n_repeats=1):
        
        self.num_workers = 4
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.n_splits = n_splits
        self.kfold = StratifiedKFold(n_splits=self.n_splits, random_state=42,
                                    shuffle=True)
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.percentage = percentage
        self.model_tag = model_tag
        self.et_range = et_range
        self.eta_range = eta_range
        self.drive_path = '/eos/user/j/jlieberm/photonRinger/datasets/notIso'
        self.all_y_preds_list = []
        self.all_y_true_list = []
        self.et = None
        self.eta = None
        self.n_repeats = n_repeats
        self.full_dataset: Dataset

        self.debug = debug
        if self.debug:
            log.warning("#### EXECUTING ON DEBUG MODE. ####")
            self.epochs = 2

    def initModel(self):
        log.info("Initializing model")
        model = get_model(self.model_tag, self.input_dim)
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(self.device)
        return model

    def initOptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def initLossFunction(self):
        # return nn.BCELoss()
        return nn.MSELoss()

    def initDataLoader(self, data, labels):
        features_tensor = torch.from_numpy(data).float()
        labels_tensor = torch.from_numpy(labels).float().view(-1, 1)

        dataset = TensorDataset(features_tensor, labels_tensor)
        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.use_cuda,
                                shuffle=True)
        return dataloader

    def main(self, index: int):
        data, target, path = self.full_dataset[index]
        test_data, train_cross_validation = self.generate_folds_with_holdout(data, target)
        self.all_training_results = []
        for fold_idx, (train_index, val_index) in enumerate(train_cross_validation):
            train_dl = self.initDataLoader(data[train_index],
                                           target[train_index])
            val_dl = self.initDataLoader(data[val_index],
                                         target[val_index])
            for repeat in range(self.n_repeats):
                log.info(f'Executing fold: {fold_idx+1}. Repeat: {repeat+1}')
                self.model = self.initModel()
                self.optimizer = self.initOptimizer()
                self.loss = self.initLossFunction()
                sp_tracker = SPCallbackPyTorch(patience=25, verbose=True)
                fold_history = {
                    'train_loss': [], 
                    'train_acc': [],
                    'val_loss': [], 
                    'val_acc': [],
                    'callbackMetrics': [],
                    'reapet': []
                }
                for epoch_ndx in range(1, self.epochs + 1):
                    avg_train_loss, avg_train_acc = self.doTraining(epoch_ndx, train_dl)

                    self.all_y_preds_list = [] 
                    self.all_y_true_list = []
                    avg_val_loss, avg_val_acc = self.doValidation(epoch_ndx, val_dl)
                    stop_training, callbackMetrics = sp_tracker.on_epoch_end(self.model,
                                                                            epoch_ndx,
                                                                            self.all_y_true_list, 
                                                                            self.all_y_preds_list)
                    fold_history['train_loss'].append(avg_train_loss)
                    fold_history['train_acc'].append(avg_train_acc)
                    fold_history['val_loss'].append(avg_val_loss)
                    fold_history['val_acc'].append(avg_val_acc)

                    if stop_training:
                        log.info(f"Fold {fold_idx}: Early stopping acionado na época {epoch_ndx}.")
                        break

                fold_history['callbackMetrics'] = callbackMetrics
                fold_history['reapet'] = repeat+1
                best_weights_for_this_run = sp_tracker.get_best_model_weights()
                best_sp_for_this_run = sp_tracker.get_best_sp_value()
                best_fa_for_this_run = sp_tracker.get_best_fa_at_knee()
                best_pd_for_this_run = sp_tracker.get_best_pd_at_knee()

                if best_weights_for_this_run is not None:
                    self.all_training_results.append({
                        'file_path': path, 
                        'fold': fold_idx,
                        'best_sp_value': best_sp_for_this_run,
                        'best_fa_value': best_fa_for_this_run,
                        'best_pd_value': best_pd_for_this_run,
                        'best_weights': best_weights_for_this_run,
                        'history': fold_history 
                    })
        return path

    def doTraining(self, epoch_ndx: int, train_dl: DataLoader):
        log.info(f"Starting training epoch {epoch_ndx}...")
        self.model.train() 

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        batch_iter = enumerate(train_dl)
        for _, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var, corrects_batch = self.computeBatchLoss(
                batch_tup,
                validation_step=False 
            )
            loss_var.backward() 
            self.optimizer.step()
            
            batch_size = batch_tup[0].size(0)
            running_loss += loss_var.item() * batch_size
            running_corrects += corrects_batch
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        log.info(f"Training: {epoch_loss = }\t{epoch_acc = }")
        return epoch_loss, epoch_acc


    def doValidation(self, epoch_ndx: int, val_dl: DataLoader):
        log.info(f"Starting validation epoch {epoch_ndx}...")
        self.model.eval() 
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad(): 
            for _, batch_tup in enumerate(val_dl):
                
                loss_var, corrects_batch = self.computeBatchLoss(
                    batch_tup,
                    validation_step=True 
                )
                
                batch_size = batch_tup[0].size(0)
                running_loss += loss_var.item() * batch_size
                running_corrects += corrects_batch
                total_samples += batch_size
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        log.info(f"Validation: {epoch_loss = }\t{epoch_acc = }")
        return epoch_loss, epoch_acc


    def computeBatchLoss(self, batch_tup: tuple[torch.Tensor, torch.Tensor],
                         validation_step: bool = False):
        
        feature, target = batch_tup
        feature = feature.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        pred_target = self.model(feature)
        
        loss = self.loss(pred_target, target) 
        
        preds_label = (pred_target >= 0.5).float()
        corrects_batch = (preds_label == target).sum().item()

        if validation_step:
            self.all_y_preds_list.append(pred_target.cpu().detach().numpy())
            self.all_y_true_list.append(target.cpu().detach().numpy())
            
        return loss, corrects_batch


    def save_results(self, folder_path: str, et: int, eta: int) -> None:
        all_training_results_template = get_results_file_name(et, eta)
        pd.DataFrame(self.all_training_results).to_pickle(
            os.path.join(folder_path, 
                         all_training_results_template))
    
    def run(self) -> None:
        if self.folder_path == None: 
            folderTemplateName = "model{model_tag}.dim{input_dim}.folds{folds}_id{id}".format(
                input_dim = self.percentage,
                model_tag = self.model_tag,
                folds=self.n_splits, 
                id=datetime.now().strftime("%Y%m%d%H%M%S"))
            self.folder_path = str(create_folder(folderTemplateName))
            
        eta_et_region = list(ParameterGrid({'eta': self.eta_range, 'et': self.et_range}))
        data_folder = [os.path.join(self.drive_path, file) for file in os.listdir(self.drive_path)
                        if (file.endswith(".npz") and 
                            file.startswith("mc23_13TeV"))
                            ]
        self.full_dataset = EgammaNpzDataset(data_folder, 
                                             percentage=self.percentage)
        self.input_dim = self.full_dataset.get_model_dim()
        for index, file in enumerate(data_folder):
            self.et, self.eta = get_et_eta(file)
            if {'eta': int(self.eta), 'et':int(self.et)} in eta_et_region:
                if verify_results(self.folder_path, self.et, self.eta):
                    self.main(index)
                    self.save_results(self.folder_path,self.et, self.eta)
                if self.debug:
                    break

    def generate_folds_with_holdout(self, features, labels, test_fold_idx=0):
        all_folds = [test_idx for _, test_idx in self.kfold.split(features, labels)]
        log.info(f"{len(all_folds)} folds were generated.")
        if not (0 <= test_fold_idx < self.n_splits):
            raise ValueError(f"test_fold_idx must be between 0 and {self.n_splits - 1}")

        log.info(f"Using fold {test_fold_idx + 1} as test set")
        holdout_indices = all_folds[test_fold_idx]

        remaining_folds = [
            fold for index, fold in enumerate(all_folds) if index != test_fold_idx
        ]
        cv_iterable = []
        for index in range(len(remaining_folds)):
            val_indices = remaining_folds[index]

            train_folds = [fold for j, fold in enumerate(remaining_folds) if j != index]
            train_indices = np.concatenate(train_folds)

            cv_iterable.append((train_indices, val_indices))

        return holdout_indices, cv_iterable