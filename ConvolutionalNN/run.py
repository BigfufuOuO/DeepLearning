import torch
from Convolutional import ImagesClassifierModel
from DataLoad import load_data
from Model import train_model
from Plots import PlotsAndLogs
import time

num_epochs = 1
learning_rate = 0.001

model = ImagesClassifierModel()
train_loader, val_loader, test_loader = load_data(batch_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ploter = PlotsAndLogs()

start_time = time.perf_counter()
train_losses, val_losses, train_accuracies, val_accuracies, true_label, predicted_label = train_model(model, optimizer, train_loader, val_loader, num_epochs)
end_time = time.perf_counter()
ploter.plot_loss_accuracy(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies)
ploter.plot_acc_matrix(true_label, predicted_label)
ploter.caculate_class_accuracy(true_label, predicted_label)
ploter.output_logs(num_epochs, learning_rate, model, 
                   train_losses, val_losses, train_accuracies, val_accuracies,
                   end_time - start_time)

