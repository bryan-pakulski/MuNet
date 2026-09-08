import numpy as np
import munet as mu
from munet import nn


class TinyUNet(nn.Module):
    def __init__(self, width=4, seed=7):
        if width <= 0:
            raise ValueError("width must be positive")
        rng = np.random.default_rng(seed)
        self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, rng=rng), nn.ReLU(),
                                     nn.Conv2d(width, width, 3, padding=1, rng=rng), nn.ReLU())
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(width, width * 2, 3, padding=1, rng=rng), nn.ReLU(),
                                        nn.Conv2d(width * 2, width * 2, 3, padding=1, rng=rng), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(width * 3, width, 3, padding=1, rng=rng), nn.ReLU(),
                                     nn.Conv2d(width, 1, 1, rng=rng))

    def forward(self, images):
        skip = self.encoder(images)
        low = self.bottleneck(self.pool(skip))
        up = nn.functional.interpolate(low, size=skip.shape[-2:], mode="nearest")
        return self.decoder(mu.cat([skip, up], dim=1))


def mask_loss(logits, targets):
    probabilities = logits.sigmoid()
    axes = (1, 2, 3)
    intersection = (probabilities * targets).sum(axes)
    dice = (2 * intersection + 1) / (probabilities.sum(axes) + targets.sum(axes) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + 1 - dice.mean()
