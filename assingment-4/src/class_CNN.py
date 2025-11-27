import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        input_shape = (channels, height, width)
        num_classes = number of output classes
        """
        super(CNN, self).__init__()

        channels, H, W = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # COMPUTE FLATTEN SIZE AUTOMATICALLY
        with torch.no_grad():
            dummy = torch.zeros(1, channels, H, W)
            flatten_size = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)