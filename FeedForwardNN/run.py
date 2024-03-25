import numpy as np
import torch
import sys
import torch.nn as nn
from FeedForward import Feed_Forward_Network, plot_fig
from NumGenerate import Num_SampleSize, Data_Process
from datetime import datetime

input_size = 1
hidden_size = 64
output_size = 1
num_hidden_layers = 5
leaning_rate = 0.001
num_epoch = 5000
batch_size = 64

model = Feed_Forward_Network(input_size, hidden_size, output_size, num_hidden_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)

train_loader, val_loader, test_loader = Data_Process(Num_SampleSize, batch_size)


# train
for epoch in range(num_epoch):
    for input, target in train_loader:
        output = model.forward(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# validation
if len(sys.argv) < 2:
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
        print(f"VAL:val_MSE_mean={val_MSE_mean}")
        
        time = datetime.now().strftime('%m%d%H%M')
        with open(f'FeedForwardNN/log/log_val_N={Num_SampleSize}.txt', 'a') as file:
            file.write("***********************\n")
            file.write(f"Model_N={Num_SampleSize}_Time={time}:\n")
            file.write(f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_hidden_layers={num_hidden_layers},\n")
            file.write(f"learning_rate={leaning_rate}, num_epoch={num_epoch}, batch_size={batch_size}:")
            file.write(f"val_MSE_mean={val_MSE_mean}\n")

        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
        y_pred = np.concatenate(y_pred)
        fig_filename = f'FeedForwardNN/fig/fig_val_N={Num_SampleSize}_Time={time}.png'
        plot_fig(x_val, y_val, y_pred, fig_filename)

elif len(sys.argv)>=2 and sys.argv[1] == "--test":
    with torch.no_grad():
        val_losses = []
        x_val = []
        y_val = []
        y_pred = []
        for input, target in test_loader:
            x_val.append(input.numpy())
            y_val.append(target.numpy())
            val_output = model(input)
            y_pred.append(val_output.numpy())
            val_loss = criterion(val_output, target)
            val_losses.append(val_loss.item())
        val_MSE_mean = np.mean(val_losses)
        print(f"TEST:Test_MSE_mean={val_MSE_mean}")
        
        time = datetime.now().strftime('%m%d%H%M')
        with open(f'FeedForwardNN/log/log_test_N={Num_SampleSize}.txt', 'a') as file:
            file.write("***********************\n")
            file.write(f"Model_N={Num_SampleSize}_Time={time}:\n")
            file.write(f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_hidden_layers={num_hidden_layers},\n")
            file.write(f"learning_rate={leaning_rate}, num_epoch={num_epoch}, batch_size={batch_size}:")
            file.write(f"val_MSE_mean={val_MSE_mean}\n")

        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
        y_pred = np.concatenate(y_pred)
        fig_filename = f'FeedForwardNN/fig/fig_test_N={Num_SampleSize}_Time={time}.png'
        plot_fig(x_val, y_val, y_pred, fig_filename)

else:
    print("Invalid input.")