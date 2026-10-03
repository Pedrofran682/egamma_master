import unittest
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from src.Parser.DynamicConfiguration import DynamicConfiguration
from src.utils import get_instance


class TestDynamicConfiguration(unittest.TestCase):
    def test_get_instance_instantiates_class(self):
        config = DynamicConfiguration(
            module="torch.nn",
            object_name="BCELoss",
            parameters={},
        )
        instance = get_instance(config)
        self.assertIsInstance(instance, nn.BCELoss)

    def test_get_instance_with_parameters(self):
        config = DynamicConfiguration(
            module="sklearn.model_selection",
            object_name="StratifiedKFold",
            parameters={"n_splits": 5, "shuffle": True, "random_state": 42},
        )
        instance = get_instance(config)
        self.assertIsInstance(instance, StratifiedKFold)
        self.assertEqual(instance.get_n_splits(), 5)

    def test_get_instance_raises_for_invalid_module(self):
        config = DynamicConfiguration(
            module="non_existent_module_xyz",
            object_name="SomeClass",
            parameters={},
        )
        with self.assertRaises(ValueError):
            get_instance(config)

    def test_get_instance_raises_for_invalid_class(self):
        config = DynamicConfiguration(
            module="torch.nn",
            object_name="NonExistentLossClass",
            parameters={},
        )
        with self.assertRaises(ValueError):
            get_instance(config)


if __name__ == "__main__":
    unittest.main()
