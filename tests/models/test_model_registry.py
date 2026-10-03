import pytest
import torch
import torch.nn as nn

from src.Models.ModelRegistry import ModelRegistry
from src.Models.Models import get_model


class TestModelRegistry:
    """Test suite for ModelRegistry and get_model."""

    def test_auto_discovery_registers_all_models_by_class_name(self) -> None:
        """Verify that all models in src/Models/egamma are discovered by class name."""
        models = ModelRegistry.list_models()
        expected_classes = [
            "ModelV1",
            "ModelV1_tanh",
            "ModelV2",
            "ModelV3",
            "ModelV4",
            "ModelV5",
            "Run2_ModelV1",
            "Run2_ModelV1_2",
        ]
        for expected in expected_classes:
            assert expected in models
            assert ModelRegistry.contains(expected)

    def test_create_models_by_class_name(self) -> None:
        """Verify instantiation via ModelRegistry.create using class name."""
        model = ModelRegistry.create("ModelV1", input_dim=50)
        assert isinstance(model, nn.Module)
        x = torch.randn(2, 50)
        out = model(x)
        assert out.shape == (2, 1)

    def test_create_models_via_get_model(self) -> None:
        """Verify backward compatibility of get_model with class names and tags."""
        model_by_class = get_model("ModelV5", input_dim=50)
        assert isinstance(model_by_class, nn.Module)

        # Legacy short tag fallback
        model_by_tag = get_model("V5", input_dim=50)
        assert isinstance(model_by_tag, nn.Module)

    def test_unknown_model_raises_value_error(self) -> None:
        """Verify that an unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="not found in ModelRegistry"):
            ModelRegistry.get_class("NonExistentModel")

    def test_explicit_registration_with_decorator(self) -> None:
        """Verify manual registration of a new model class."""
        @ModelRegistry.register
        class CustomTestModel(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        assert "CustomTestModel" in ModelRegistry.list_models()
        instance = ModelRegistry.create("CustomTestModel", input_dim=10)
        assert isinstance(instance, CustomTestModel)
