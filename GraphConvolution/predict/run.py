import torch
from DataLoad import DataProcessor
from LinkPredict import GCNNet
from TrainModel import train_model
from Recorder import Recorder

Learning_rate = 0.01
Weight_decay = 5e-4
Epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:" + device.type)

dataset = DataProcessor('cora')
model = GCNNet(dataset._data.features.shape[1]).to(device)
criterion = torch.nn.functional.binary_cross_entropy_with_logits
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=Weight_decay)
shceduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, 
                                                       threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

torch_dataset = dataset.Transform_Data()
model = train_model(Epochs, model, criterion, optimizer, shceduler, torch_dataset, dataset)
recorder = Recorder()
recorder.Output_parameters(Learning_rate, Weight_decay, model.drop_rate, Epochs, dataset.dataset)