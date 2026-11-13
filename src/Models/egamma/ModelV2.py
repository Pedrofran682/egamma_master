import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelV2(nn.Module):
    """1D Convolutional Neural Network with 64 and 32 feature channels."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the ModelV2 network layers.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.input_dim: int = input_dim
        self.conv1: nn.Conv1d = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=2, padding="same"
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            in_channels=64, out_channels=32, kernel_size=2, padding="same"
        )
        self.fc1_in_features: int = 32 * input_dim
        self.fc1: nn.Linear = nn.Linear(self.fc1_in_features, input_dim)
        self.fc2: nn.Linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward computation graph.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output sigmoid probability tensor of shape (batch_size, 1).
        """
        x = x.view(-1, 1, self.input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
