import torch
import torch.nn as nn
import torch.nn.functional as F


class Run2_ModelV1(nn.Module):
    """Run 2 legacy architecture V1 with sigmoid output activation."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the Run2_ModelV1 network layers.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_dim, input_dim)
        self.fc2: nn.Linear = nn.Linear(input_dim, 8)
        self.fc3: nn.Linear = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass with ReLU hidden activations and sigmoid output.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output prediction tensor in range (0, 1).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
