import torch
import torch.nn as nn
import torch.nn.functional as F


class Run2_ModelV1(nn.Module):
    def __init__(self, input_dim):
        super(Run2_ModelV1, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
