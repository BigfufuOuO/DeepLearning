import torch.nn as nn

class Feed_Forward_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feed_Forward_Network, self).__init__()
        self.fully_connected1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fully_connected2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fully_connected1(x)
        out = self.relu(out)
        out = self.fully_connected2(out)
        return out
