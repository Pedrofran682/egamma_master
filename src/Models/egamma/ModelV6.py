import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelV6(nn.Module):
    """Multi-layer Perceptron (dense network) with three linear projection layers."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the ModelV6 network layers.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, input_dim)
        self.fc2: nn.Linear = nn.Linear(input_dim, 8)
        self.fc3: nn.Linear = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward computation graph.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output sigmoid probability tensor of shape (batch_size, 1).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
