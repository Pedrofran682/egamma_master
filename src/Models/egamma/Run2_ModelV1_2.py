import torch
import torch.nn as nn
import torch.nn.functional as F


class Run2_ModelV1_2(nn.Module):
    """Run 2 legacy architecture V1 variant with hyperbolic tangent output."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the Run2_ModelV1_2 network layers.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, input_dim)
        self.fc2: nn.Linear = nn.Linear(input_dim, 8)
        self.fc3: nn.Linear = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass with ReLU hidden activations and tanh output.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output prediction tensor in range (-1, 1).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
