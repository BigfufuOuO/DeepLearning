import torch
from Convolutional import ImagesClassifierModel
from DataLoad import load_data
from Model import train_model
from Plots import Plots


model = ImagesClassifierModel()
train_loader, val_loader, test_loader = load_data(batch_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
ploter = Plots()


train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, optimizer, train_loader, val_loader, num_epochs)
ploter.plot_loss(num_epochs, train_losses, val_losses)
ploter.plot_accuracy(num_epochs, train_accuracies, val_accuracies)
