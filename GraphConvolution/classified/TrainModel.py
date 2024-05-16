import os
import torch
from DataLoad import Data
from GCNNet import GCNNet
from Recorder import Recorder
from tqdm import tqdm

def Save_Model(model, acc_val_final, model_name):
    print('\n')
    print(f'Validation accuracy: {acc_val_final:.4f}')
    # get the current path
    current_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(current_path, 'models')
    if model_name == 'cora':
        if acc_val_final > 0.8:
            torch.save(model.state_dict(), f'{model_path}/GCNNet_{model_name}_acc{acc_val_final:.5f}.pth')
            print(f'Model saved in {model_path}/GCNNet_{model_name}_acc{acc_val_final:.5f}.pth')
    elif model_name == 'citeseer':
        if acc_val_final > 0.7:
            torch.save(model.state_dict(), f'{model_path}/GCNNet_{model_name}_acc{acc_val_final:.5f}.pth')
            print(f'Model saved in {model_path}/GCNNet_{model_name}_acc{acc_val_final:.5f}.pth')

def validation_loss(model, criterion, dataset):
    model.eval()
    with torch.no_grad():
        output = model(dataset.features, dataset.Matrix_sparse)
        loss = criterion(output[dataset.val_mask], dataset.labels[dataset.val_mask].long())
    return loss.cpu().numpy().item()
    
def test_acc(model, mask, dataset):
    model.eval()
    with torch.no_grad():
        output = model(dataset.features, dataset.Matrix_sparse)
        pred = output[mask].max(1)[1]
        acc = torch.eq(pred, dataset.labels[mask]).float().mean()
    return acc.cpu().numpy().item()
    
    
def train_model(epochs, model, criterion, optimizer, scheduler, dataset, origial_dataset_name):
    '''
    Input:
        dataset: torch tensor data.
    '''
    recorder = Recorder()
    array_train_loss = []
    array_val_loss = []
    array_train_acc = []
    array_val_acc = []
    learning_rate = []
    
    # start training
    
    # process bar
    process_bar = tqdm(total=epochs, position=0, leave=True, ncols=100)
    for epoch in range(epochs):
        model.train()
        output = model(dataset.features, dataset.Matrix_sparse)
        # supervise training
        train_mask_output = output[dataset.train_mask]
        
        # calculate loss
        loss = criterion(train_mask_output, dataset.labels[dataset.train_mask].long())
        train_loss = loss.detach().cpu().numpy().item()
        optimizer.zero_grad()
        loss.backward()
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
        process_bar.update(1)
        # scheduler.step(val_loss)
        learning_rate.append(optimizer.param_groups[0]['lr'])
    
    process_bar.close()
    # Output the result
    recorder.Plot_loss(epochs, array_train_loss, array_val_loss, array_train_acc, array_val_acc, learning_rate, origial_dataset_name)
    recorder.Output_logs(array_train_loss, array_val_loss, array_train_acc, array_val_acc, origial_dataset_name)
    Save_Model(model, array_val_acc[-1], origial_dataset_name)
    return model