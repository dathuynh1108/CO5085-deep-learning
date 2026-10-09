"""E1: three classifiers trained from random initialization."""
from torch import Tensor, nn


class SoftmaxClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def forward(self, x: Tensor) -> Tensor:
        # CrossEntropyLoss performs log-softmax; return logits, not probabilities.
        return self.classifier(x)


class MLPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))
