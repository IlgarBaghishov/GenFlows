import torch.nn as nn


class CNN3D(nn.Module):
    """3D CNN regressor for predicting lobe properties from binary volumes.

    Takes binary facies volumes (B, 1, 50, 50, 50) and predicts 5 properties:
    [height, radius, aspect_ratio, sin(2*angle*pi/180), cos(2*angle*pi/180)].

    Architecture: 5 Conv3d layers (1->64->64->64->128->128) with MaxPool3d
    after 2nd, 4th, 5th convolutions, followed by FC layers (27648->512->256->256->128->5).
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(128),

            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
