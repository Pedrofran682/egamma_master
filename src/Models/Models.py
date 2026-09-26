import torch.nn as nn

from src.Models.ModelRegistry import ModelRegistry


def get_model(tag: str, input_dim: int) -> nn.Module:
    """Instantiate a neural network model matching the requested architecture tag.

    Delegates to ModelRegistry which automatically discovers all model classes.

    Args:
        tag: Model architecture identifier string (class name or tag).
        input_dim: Number of input features (rings).

    Returns:
        Configured PyTorch nn.Module instance.
    """
    return ModelRegistry.create(tag, input_dim=input_dim)
