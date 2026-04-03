"""
Model definition: SmallCNN for CIFAR-10.
"""

import torch.nn as nn
import torch.nn.functional as F
# 
# class SmallCNN(nn.Module):
#     """
#     Small convolutional network for CIFAR-10 (32x32 images).
#  
#     Architecture:
#         Conv(3->32)   -> ReLU -> MaxPool(2)   [32->16]
#         Conv(32->64)  -> ReLU -> MaxPool(2)   [16->8]
#         Conv(64->128) -> ReLU -> MaxPool(2)   [8->4]
#         FC(2048->256) -> ReLU -> FC(256->n_classes)
#  
#     Args:
#         n_classes: number of output classes (default: 10 for CIFAR-10)
#     """
#  
#     def __init__(self, n_classes: int = 10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool  = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.fc1   = nn.Linear(128 * 4 * 4, 256)
#         self.fc2   = nn.Linear(256, n_classes)
# 
#     def forward(self, x):
#         """
#         Args:
#             x: input tensor, shape (B, 3, 32, 32)
# 
#         Returns:
#             logits tensor, shape (B, n_classes)
#         """
#         x = self.pool(F.relu(self.conv1(x)))   # 32->16
#         x = self.pool(F.relu(self.conv2(x)))   # 16->8
#         x = self.pool(F.relu(self.conv3(x)))   # 8->4
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 32->16->8
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8->4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
