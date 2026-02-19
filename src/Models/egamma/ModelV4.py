import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelV4(nn.Module):
    def __init__(self, input_dim):
        super(ModelV4, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=4, padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=3, padding="same"
        )
        self.fc1 = nn.Linear(in_features=16 * input_dim, out_features=input_dim)
        self.fc2 = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * self.input_dim)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
