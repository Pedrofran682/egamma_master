import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelV1_tanh(nn.Module):
    def __init__(self, input_dim):
        super(ModelV1_tanh, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=8, kernel_size=2, padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=8, out_channels=4, kernel_size=2, padding="same"
        )
        self.fc1_in_features = 4 * input_dim
        self.fc1 = nn.Linear(self.fc1_in_features, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
