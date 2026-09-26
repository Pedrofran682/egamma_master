import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelV1_tanh(nn.Module):
    """1D Convolutional Neural Network with tanh activations on intermediate layers."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the ModelV1_tanh network layers.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.input_dim: int = input_dim
        self.conv1: nn.Conv1d = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=2, padding="same"
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            in_channels=8, out_channels=4, kernel_size=2, padding="same"
        )
        self.fc1_in_features: int = 4 * input_dim
        self.fc1: nn.Linear = nn.Linear(self.fc1_in_features, input_dim)
        self.fc2: nn.Linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward computation graph with tanh activations.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output sigmoid probability tensor of shape (batch_size, 1).
        """
        x = x.view(-1, 1, self.input_dim)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
