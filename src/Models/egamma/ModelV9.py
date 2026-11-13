import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelV9(nn.Module):
    """Single hidden layer Multi-layer Perceptron with 8 units."""

    def __init__(self, input_dim: int) -> None:
        """Initialize ModelV9 layers.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, 8)
        self.fc2: nn.Linear = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output sigmoid probability tensor of shape (batch_size, 1).
        """
        x = F.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

