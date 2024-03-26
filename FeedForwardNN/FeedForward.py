import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


class Feed_Forward_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(Feed_Forward_Network, self).__init__()
        # self.fully_connected1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.relu = nn.ReLU()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fully_connected2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out = self.fully_connected1(x)
        out = x
        for layer in self.hidden_layers:
            out = self.relu(layer(out))
        out = self.fully_connected2(out)
        return out

def plot_fig(x_val, y_val, y_pred, output_file):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_val, y_pred, label='Predited Points', color='blue', alpha=0.5)
    # plt.scatter(x_val, y_val, label='Original Points', color='red', alpha=0.5)
    x = np.linspace(1, 16, 100)
    y = np.log2(x) + np.cos(np.pi * x / 2)
    plt.plot(x, y, label='True cure', color='black', alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(' Predicted Point and Origin curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()