import torch
from DataLoad import DataProcessor
from LinkPredict import GCNNet
import os, sys
from TrainModel import generate_true_labels
from sklearn.metrics import roc_auc_score
# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'cora'
dataset = DataProcessor(dataset_name)
torch_dataset = dataset.Transform_Data()
model = GCNNet(dataset._data.features.shape[1]).to(device)
filenames = os.listdir(os.path.join(os.path.dirname(__file__), 'models'))
maxacc = 0
for filename in filenames: 
    if dataset_name in filename:
        acc = float(filename[len(dataset_name)+11:len(dataset_name)+18])
        maxacc = max(maxacc, acc)
model_file = f'GCNNet_{dataset_name}_auc{maxacc:.5f}.pth'
print(model_file)

model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models/') + model_file))

with torch.no_grad():
    model.eval()
    test_positive_edge_pairs = torch_dataset.Edge_pairs[torch_dataset.test_mask].t()
    negative_edge_pairs = dataset.negative_sampling(test_positive_edge_pairs, num_neg_rates=3)
    output = model(torch_dataset.features, torch_dataset.Matrix_sparse, \
                     test_positive_edge_pairs, negative_edge_pairs)
    true_labels = generate_true_labels(test_positive_edge_pairs, negative_edge_pairs)
    auc = roc_auc_score(true_labels.cpu().numpy(), output.cpu().numpy())
    print(f'Test AUC: {auc:.4f}')