import os
import torch
from DataLoad import Data
from LinkPredict import GCNNet
from Recorder import Recorder
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def validation_test(model, criterion, torch_dataset):
    with torch.no_grad():
        model.eval()
        val_positive_edge_pairs = torch_dataset.Edge_pairs[torch_dataset.val_mask]
        val_negative_edge_pairs = torch_dataset.negative_edge_pairs(val_positive_edge_pairs)
        output = model(torch_dataset.features, torch_dataset.Matrix_sparse, \
                       val_positive_edge_pairs, val_negative_edge_pairs)
        true_labels = generate_true_labels(val_positive_edge_pairs, val_negative_edge_pairs)
        loss = criterion(output, true_labels)
        auc = roc_auc_score(true_labels.cpu().numpy(), output.cpu().numpy())
        return loss.cpu().numpy().item(), auc
        
'''
def test_acc(model, mask, torch_dataset):
    model.eval()
    with torch.no_grad():
        output = model(torch_dataset.features, torch_dataset.Matrix_sparse)
        pred = output[mask].max(1)[1]
        acc = torch.eq(pred, torch_dataset.labels[mask]).float().mean()
    return acc.cpu().numpy().item()
'''

def generate_true_labels(posi_edge_pairs, nega_edge_pairs):
    num_pairs = posi_edge_pairs.shape[1] + nega_edge_pairs.shape[1]
    labels = torch.zeros(num_pairs, dtype=torch.long)
    labels[:posi_edge_pairs.shape[1]] = 1
    return labels.to(device)
    
    
def train_model(epochs, model, criterion, optimizer, scheduler, torch_dataset, original_dataset):
    '''
    Input:
        torch_dataset: torch tensor data.
    '''
    recorder = Recorder()
    array_train_loss = []
    array_val_loss = []
    array_train_auc = []
    array_val_auc = []
    learning_rate = []
    
    # start training
    
    # process bar
    process_bar = tqdm(total=epochs, position=0, leave=True, ncols=100)
    for epoch in range(epochs):
        model.train()
        train_positive_edge_pairs = torch_dataset.Edge_pairs[torch_dataset.train_mask]
        train_negative_edge_pairs = original_dataset.negative_edge_pairs(train_positive_edge_pairs)
        output = model(torch_dataset.features, torch_dataset.Matrix_sparse, \
                       train_positive_edge_pairs, train_negative_edge_pairs)
        
        true_lables = generate_true_labels(train_positive_edge_pairs, train_negative_edge_pairs)
        # calculate loss
        loss = criterion(output, true_lables)
        train_loss = loss.detach().cpu().numpy().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # caculate accuracy
        train_auc = roc_auc_score(true_lables.cpu().numpy(), output.cpu().numpy())
        val_loss, val_auc = validation_test(model, criterion, torch_dataset)
        
        
        # Record result down
        array_train_loss.append(train_loss)
        array_val_loss.append(val_loss)
        array_train_auc.append(train_auc)
        array_val_auc.append(val_auc)
        process_bar.update(1)
        # scheduler.step(val_loss)
        learning_rate.append(optimizer.param_groups[0]['lr'])
    
    process_bar.close()
    # Output the result
    recorder.Plot_loss_auc(epochs, array_train_loss, array_val_loss, array_train_auc, \
                            array_val_auc, learning_rate, original_dataset)
    recorder.Output_logs(array_train_loss, array_val_loss, array_train_auc, array_val_auc, original_dataset)
    Save_Model
    
    return model