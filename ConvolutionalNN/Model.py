import os
import torch
import torch.nn as nn
import numpy as np
# from Convolutional import ImagesClassifierModel


# Train the model
# model = ImagesClassifierModel()
# train_loader, val_loader, test_loader = load_data(batch_size=64)

def train_model(model, optimizer, train_loader, val_loader, num_epochs, criterion=nn.CrossEntropyLoss()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:" + device.type)
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []
    # Training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        # model.train()
        train_loss = []
        
        for inputs, labels in train_loader:
            # Move the input and target data to the proper device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(inputs)
            train_accurate = model.accuracy(labels, output)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            # loss draw
            train_loss.append(loss.item())
        
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()
            for input, labels in val_loader:
                val_loss = []
                input = input.to(device)
                labels = labels.to(device)
                
                output = model(input)
                val_accurate = model.accuracy(labels, output)
                loss = criterion(output, labels)
                
                val_loss.append(loss.item())
            val_losses_mean = np.mean(val_loss)
            val_losses.append(val_losses_mean)
            val_accuracy.append(val_accurate)

        train_losses_mean = np.mean(train_loss)
        train_losses.append(train_losses_mean)
        train_accuracy.append(train_accurate)
    
    return train_losses, val_losses, train_accuracy, val_accuracy
            