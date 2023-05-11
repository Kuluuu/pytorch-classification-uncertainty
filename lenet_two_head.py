import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet_TwoHead(nn.Module):
    def __init__(self, num_classes=10, dropout=False):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.acc_head = nn.Sequential(
            nn.Linear(28800, 500), nn.ReLU(),
            nn.Linear(500, num_classes)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(28800, 500), nn.ReLU(),
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        preds = self.acc_head(x)
        uncertainty = self.uncertainty_head(x)
        return preds, uncertainty
