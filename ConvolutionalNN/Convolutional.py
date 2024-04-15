import torch.nn as nn
import torch

# A Convolutional Neural Network
class ImagesClassifierModel(nn.Module):
    def __init__(self, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        # Define the layers
        self.kenel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=self.kenel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=self.kenel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 12 * 16 * 16
            nn.Dropout(dropout),
            
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=self.kenel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=self.kenel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 * 8 * 8
            nn.Dropout(dropout),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kenel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kenel_size, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 * 4 * 4
            nn.Dropout(dropout),
            
            nn.BatchNorm2d(128),
            nn.Flatten(), # change the shape to 1D
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        return self.network(x)

    def inaccuracy(self, label, predicted):
        # caculate the accuracy
        # label, output is tensor
        inaccurate = 0
        for i in range(len(label)):
            if predicted[i] != label[i]:
                inaccurate += 1
        return inaccurate