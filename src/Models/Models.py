import torch.nn as nn

from src.Models import egamma


def get_model(tag: str, input_dim: int) -> nn.Module:
    """Instantiate a neural network model matching the requested architecture tag.

    Args:
        tag: Model architecture identifier string.
        input_dim: Number of input features (rings).

    Returns:
        Configured PyTorch nn.Module instance.

    Raises:
        ValueError: If the tag does not match any registered architecture.
    """
    if tag == "V1":
        return egamma.ModelV1(input_dim)
    if tag == "V1_tanh":
        return egamma.ModelV1_tanh(input_dim)
    if tag == "V2":
        return egamma.ModelV2(input_dim)
    if tag == "V3":
        return egamma.ModelV3(input_dim)
    if tag == "V4":
        return egamma.ModelV4(input_dim)
    if tag == "V5":
        return egamma.ModelV5(input_dim)
    if tag == "Run2_ModelV1":
        return egamma.Run2_ModelV1(input_dim)
    if tag == "Run2_ModelV1_2":
        return egamma.Run2_ModelV1_2(input_dim)
    raise ValueError(f"Unknown or unspecified model architecture tag: '{tag}'")
