import torch.nn as nn

# A Convolutional Neural Network
class ImagesClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers
        self.kenel_size = 3
        self.padding = 0
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=6, 
                               kernel_size=self.kenel_size,
                               padding=self.padding)# 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 pooling
        self.conv2 = nn.Conv2d(in_channels=6, 
                               out_channels=10, 
                               kernel_size=self.kenel_size,
                               padding=self.padding)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        
    def forward(self, x):
        return x