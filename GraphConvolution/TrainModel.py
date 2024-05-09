import torch
from DataLoad import Data
from GCNNet import GCNNet
from Recorder import Recorder

def validation_loss(model, criterion, dataset):
    model.eval()
    with torch.no_grad():
        output = model(dataset.features, dataset.Matrix_sparse)
        loss = criterion(output[dataset.val_mask], dataset.labels[dataset.val_mask].long())
    return loss
    
def test_acc(model, mask, dataset):
    model.eval()
    with torch.no_grad():
        output = model(dataset.features, dataset.Matrix_sparse)
        pred = output[mask].max(1)[1]
        acc = torch.eq(pred, dataset.labels[mask]).float().mean()
    return acc
    
    
def train_model(epochs, model, criterion, optimizer, dataset):
    recorder = Recorder()
    array_train_loss = []
    array_val_loss = []
    array_train_acc = []
    array_val_acc = []
    
    # start training
    model.train()
    for epoch in range(epochs):
        output = model(dataset.features, dataset.Matrix_sparse)
        # supervise training
        train_mask_output = output[dataset.train_mask]
        
        # calculate loss
        train_loss = criterion(train_mask_output, dataset.labels[dataset.train_mask].long())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # caculate accuracy
        train_acc = test_acc(model, dataset.train_mask, dataset)
        val_acc = test_acc(model, dataset.val_mask, dataset)
        val_loss = validation_loss(model, criterion, dataset)
        
        # Record result down
        array_train_loss.append(train_loss)
        array_val_loss.append(val_loss)
        array_train_acc.append(train_acc)
        array_val_acc.append(val_acc)
        
    # Output the result
    recorder.Plot_loss(epochs, array_train_loss, array_val_loss, array_train_acc, array_val_acc)
    