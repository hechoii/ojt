import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
  def __init__(self, input_dim = 784, hidden1 = 256, hidden2 = 128, num_classes = 10):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden1)
    self.bn1 = nn.BatchNorm1d(hidden1)
    self.fc2 = nn.Linear(hidden1, hidden2)
    self.bn2 = nn.BatchNorm1d(hidden2)
    self.fc3 = nn.Linear(hidden2, num_classes)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Flatten: (B,1,28,28) -> (B,784)
    x = x.view(x.size(0),-1)

    # layer 1
    x = self.fc1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 2
    x = self.fc2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.dropout(x)

    # Output Layer
    x = self.fc3(x)
    return x