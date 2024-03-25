import numpy as np
import torch
import torch.nn as nn
from FeedForward import Feed_Forward_Network, plot_fig
from NumGenerate import Num_SampleSize, Data_Process
from datetime import datetime

input_size = 1
hidden_size = 64
output_size = 1
leaning_rate = 0.0001
num_epoch = 1000
batch_size = 64

model = Feed_Forward_Network(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

x_filename = f'data/data_x_N={Num_SampleSize}.npy'
y_filename = f'data/data_y_N={Num_SampleSize}.npy'
train_loader, val_loader, test_loader = Data_Process(Num_SampleSize, batch_size, x_filename, y_filename)

# train
for epoch in range(num_epoch):
    for input, target in train_loader:
        output = model.forward(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# validation
with torch.no_grad():
    val_losses = []
    x_val = []
    y_val = []
    y_pred = []
    for input, target in val_loader:
        x_val.append(input.numpy())
        y_val.append(target.numpy())
        val_output = model(input)
        y_pred.append(val_output.numpy())
        val_loss = criterion(val_output, target)
        val_losses.append(val_loss.item())
    val_MSE_mean = np.mean(val_losses)
    print(f"val_MSE_mean={val_MSE_mean}")
    
    time = datetime.now().strftime('%m%d%H%M')
    with open(f'log/log_N={Num_SampleSize}.txt', 'a') as file:
        file.write("***********************\n")
        file.write(f"Model_N={Num_SampleSize}_Time={time}:\n")
        file.write(f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, \n")
        file.write(f"learning_rate={leaning_rate}, num_epoch={num_epoch}, batch_size={batch_size}:")
        file.write(f"val_MSE_mean={val_MSE_mean}\n")

    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    y_pred = np.concatenate(y_pred)
    fig_filename = f'fig/fig_val_N={Num_SampleSize}_Time={time}.png'
    plot_fig(x_val, y_val, y_pred, fig_filename)