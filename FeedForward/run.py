import torch
import torch.nn as nn
from FeedForward import Feed_Forward_Network
from NumGenerate import Generate_dataset, Data_Process

input_size = 1
hidden_size = 32
output_size = 1
leaning_rate = 0.001
num_epoch = 1000
batch_size = 64
Num_SampleSize = 200


train_loader, val_loader, test_loader = Generate_dataset(Num_SampleSize, batch_size)
model = Feed_Forward_Network(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

# train
for epoch in range(num_epoch):
    for input, target in train_loader:
        output = model.forward(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

