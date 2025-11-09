import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
