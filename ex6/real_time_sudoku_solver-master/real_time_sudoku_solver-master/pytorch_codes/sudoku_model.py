import torch
import torch.nn as nn

class SudokuCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # index 0
            nn.ReLU(inplace=True),             # index 1
            nn.Conv2d(32, 64, kernel_size=3),  # index 2
            nn.ReLU(inplace=True),             # index 3
            nn.MaxPool2d(kernel_size=2),       # index 4
            nn.Dropout(0.25),                  # index 5
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),      # index 0
            nn.ReLU(inplace=True),             # index 1
            nn.Dropout(0.5),                   # index 2
            nn.Linear(128, 9),                 # index 3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x