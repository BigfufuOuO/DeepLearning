import torch
from DataLoad import DataProcessor
from GCNNet import GCNNet
from TrainModel import train_model

Learning_rate = 0.01
Weight_decay = 5e-4
Epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = DataProcessor('cora')
model = GCNNet(dataset.features.shape[1]).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=Weight_decay)

torch_dataset = dataset.Transform_Data()
train_model(Epochs, model, criterion, optimizer, torch_dataset)