import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.Callbacks.SPCallbackPyTorch import SPCallbackPyTorch
from src.core.Trainers.FoldTrainer import FoldTrainer
from src.core.Trainers.ResultsRecorder import ResultsRecorder
from src.core.Trainers.TrainingFactory import TrainingFactory
from src.Parser.DynamicConfiguration import DynamicConfiguration
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration


class SimpleTestModel(nn.Module):
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class TestTrainers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.features = np.random.randn(100, 10).astype(np.float32)
        self.labels = np.random.choice([0, 1], size=100).astype(np.int32)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model = SimpleTestModel(input_dim=10)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_results_recorder(self):
        recorder = ResultsRecorder()
        recorder.record(
            repeat=1,
            file_path="mock_path.npz",
            fold=0,
            best_sp=0.95,
            best_fa=0.02,
            best_pd=0.98,
            best_weights={"fc.weight": torch.tensor([1.0])},
            history={"train_loss": [0.5, 0.3]},
        )

        saved_file = recorder.save(self.temp_dir.name, et=0, eta=0, repeat=1, fold_idx=0)
        self.assertTrue(os.path.exists(saved_file))

        df = pd.read_pickle(saved_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["best_sp_value"], 0.95)
        self.assertEqual(df.iloc[0]["reapet"], 1)

        recorder.clear()
        self.assertEqual(len(recorder.records), 0)

    def test_fold_trainer_fit(self):
        dataset = TensorDataset(
            torch.from_numpy(self.features).float(),
            torch.from_numpy(self.labels).float().view(-1, 1),
        )
        dl = DataLoader(dataset, batch_size=20)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()
        device = torch.device("cpu")

        trainer = FoldTrainer(self.model, optimizer, loss_fn, device)
        callback = SPCallbackPyTorch(patience=5, verbose=False)

        results = trainer.fit(train_dl=dl, val_dl=dl, epochs=2, callback=callback)

        self.assertIn("history", results)
        self.assertEqual(len(results["history"]["train_loss"]), 2)
        self.assertEqual(len(results["history"]["val_loss"]), 2)
        self.assertIsNotNone(results["best_weights"])

    def test_training_factory_create_dataloader(self):
        config = NeuralRingerTrainerConfiguration(
            config_name="test_config",
            batch_size=16,
            num_workers=0,
            balance_data=False,
            epochs=2,
            n_initializations=1,
            et_range_idx=[0],
            eta_range_idx=[0],
            pred_target_limiar=0.5,
            loss_function=DynamicConfiguration(
                module="torch.nn", object_name="BCELoss", parameters={}
            ),
            optimizer_function=DynamicConfiguration(
                module="torch.optim", object_name="Adam", parameters={"lr": 0.01}
            ),
            kFold=DynamicConfiguration(
                module="sklearn.model_selection",
                object_name="StratifiedKFold",
                parameters={"n_splits": 5},
            ),
            model=DynamicConfiguration(
                module="tests.test_trainers",
                object_name="SimpleTestModel",
                parameters={"input_dim": 10},
            ),
            dataset=DynamicConfiguration(
                module="src.core.Datasets.EgammaNpzDataset",
                object_name="EgammaNpzDataset",
                parameters={"drive_path": "data/"},
            ),
        )
        factory = TrainingFactory(config, device=torch.device("cpu"))
        dl = factory.create_dataloader(self.features, self.labels)

        self.assertIsInstance(dl, DataLoader)
        batch_x, batch_y = next(iter(dl))
        self.assertEqual(batch_x.shape[0], 16)
        self.assertEqual(batch_x.shape[1], 10)


if __name__ == "__main__":
    unittest.main()
