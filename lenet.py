import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, feature_dim=64, in_channels=1):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2))
        self.fc = nn.Linear(50 * 4 * 4, feature_dim)

    def forward(self, X):
        assert X.size(2) == 28 and X.size(3) == 28

        X = self.conv1(X)
        X = self.conv2(X)
        X = X.view(-1, 50 * 4 * 4)
        X = self.fc(X)
        return X

