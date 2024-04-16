import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# from Convolutional import ImagesClassifierModel


# Train the model
# model = ImagesClassifierModel()
# train_loader, val_loader, test_loader = load_data(batch_size=64)
def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs, criterion=nn.CrossEntropyLoss()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:" + device.type)
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []
    learning_rates_array = []
    # Training loop
    for epoch in range(num_epochs):
        process_bar = tqdm(total=len(train_loader),
                           desc=f'Epoch {epoch + 1}/{num_epochs}', 
                           unit='batch', position=0, leave=True,
                           ncols=100)
        true_label = []
        predict_label = []
        # Set the model to training mode
        model.train()
        num_inaccurate_train = 0
        train_loss = []
        
        for inputs, labels in train_loader:
            # Move the input and target data to the proper device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(inputs)
            _, predicted = torch.max(output, 1)
            train_inaccurate = model.inaccuracy(labels, predicted)
            num_inaccurate_train += train_inaccurate
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
        
            
            # loss draw
            train_loss.append(loss.item())
            process_bar.update(1)
        train_losses_mean = np.mean(train_loss)
        train_losses.append(train_losses_mean)
        train_accuracy.append(1 - num_inaccurate_train / len(train_loader.dataset))
        
        with torch.no_grad():
            # Set the model to evaluation mode
            num_inaccurate_val = 0
            model.eval()
            for input, labels in val_loader:
                val_loss = []
                input = input.to(device)
                labels = labels.to(device)
                
                output = model(input)
                _, predicted = torch.max(output, 1)
                val_inaccurate = model.inaccuracy(labels, predicted)
                num_inaccurate_val += val_inaccurate
                
                loss = criterion(output, labels)
                
                val_loss.append(loss.item())
                if epoch == num_epochs - 1:
                    for i in range(len(predicted)):
                        predict_label.append(predicted[i].item())
                    for i in range(len(labels)):
                        true_label.append(labels[i].item())
            val_losses_mean = np.mean(val_loss)
            val_losses.append(val_losses_mean)
            val_accuracy.append(1 - num_inaccurate_val / len(val_loader.dataset))

        process_bar.close()
        scheduler.step(val_losses_mean)
        learning_rates_array.append(optimizer.param_groups[0]['lr'])
        
        print('train loss: {:.4f}, val loss: {:.4f}, train accuracy: {:.4f}, val accuracy: {:.4f}'.format(train_losses_mean, val_losses_mean, train_accuracy[-1], val_accuracy[-1]))
        print('leanring rate: {}'.format(optimizer.param_groups[0]['lr']))
    
    return train_losses, val_losses, train_accuracy, val_accuracy, true_label, predict_label, learning_rates_array
            