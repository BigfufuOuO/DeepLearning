import torch
from DataLoad import DataProcessor
from GCNNet import GCNNet
import os, sys
from TrainModel import test_acc
# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'citeseer'
dataset = DataProcessor(dataset_name)
torch_dataset = dataset.Transform_Data()
model = GCNNet(dataset._data.features.shape[1], drop_rate=0).to(device)
filenames = os.listdir(os.path.join(os.path.dirname(__file__), 'models'))
maxacc = 0
for filename in filenames: 
    if dataset_name in filename:
        acc = float(filename[len(dataset_name)+11:len(dataset_name)+18])
        maxacc = max(maxacc, acc)
model_file = f'GCNNet_{dataset_name}_acc{maxacc:.5f}.pth'
print(model_file)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models/') + model_file))

# test
with torch.no_grad():
    model.eval()
    acc = test_acc(model, torch_dataset.test_mask, torch_dataset)
    print(f'Test accuracy: {acc:.4f}')
