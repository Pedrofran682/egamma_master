import torch.nn as nn
import torch.nn.functional as F
import torch


class ModelV1(nn.Module):
    def __init__(self, input_dim):
        super(ModelV1, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=4, 
                               kernel_size=2, 
                               padding='same') 
        self.conv2 = nn.Conv1d(in_channels=4, 
                               out_channels=8, 
                               kernel_size=2, 
                               padding='same')
        self.fc1_in_features = 8 * input_dim
        self.fc1 = nn.Linear(self.fc1_in_features, 
                             input_dim)
        self.fc2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class ModelV3(nn.Module):
    def __init__(self, input_dim):
        super(ModelV3, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=32, 
                               kernel_size=2, 
                               padding='same' )
        self.conv2 = nn.Conv1d(in_channels=32, 
                               out_channels=16, 
                               kernel_size=2, 
                               padding='same')
        self.fc1 = nn.Linear(in_features=16 * input_dim, 
                             out_features=input_dim)
        self.fc2 = nn.Linear(in_features=input_dim, 
                             out_features=1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * self.input_dim)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class ModelV5(nn.Module):
    
    def __init__(self, input_dim):
        super(ModelV5, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    

def get_model(tag: str, input_dim: int) -> nn.Module:
    if tag == "V1":
        return ModelV1(input_dim)
    if tag == "V3":
        return ModelV3(input_dim)
    if tag == "V5":
        return ModelV5(input_dim)