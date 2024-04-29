import torch
import torch.nn as nn


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.device = self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸减半
            # layer 2
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸再减半
            # layer 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸再减半
            # layer 4
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸再减半
        )

    def forward(self, x):
        # x = x.to(self.device)
        x = self.features(x)
        return x
