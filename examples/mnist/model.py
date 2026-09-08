import numpy as np
from munet import nn


class DigitCNN(nn.Module):
    def __init__(self, width=8, seed=7):
        if width <= 0:
            raise ValueError("width must be positive")
        rng = np.random.default_rng(seed)
        self.conv1 = nn.Conv2d(1, width, 3, padding=1, rng=rng)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1, rng=rng)
        self.pool = nn.AvgPool2d(2)
        self.head = nn.Linear(width * 2 * 7 * 7, 10, rng=rng)

    def forward(self, images):
        x = self.pool(self.conv1(images).relu())
        x = self.pool(self.conv2(x).relu())
        return self.head(x.flatten(1))
