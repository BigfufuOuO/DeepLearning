import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data
from torch_geometric.datasets import Planetoid
from collections import namedtuple

Data = namedtuple('Data', ['Matrix_sparse', 'Matrix_degree', 'features', 'labels', \
                        'train_mask', 'val_mask', 'test_mask'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataProcessor:
    def __init__(self, dataset='cora'):
        '''
        Input:
            dataset: str, the name of dataset
        
        Member:
            Matrix_sparse: scipy.sparse.coo_matrix, the sparse matrix of adjacency matrix
            Matrix_degree: numpy.ndarray, the degree matrix
            features: numpy.ndarray, the feature matrix
            labels: numpy.ndarray, the label matrix
            train_mask: numpy.ndarray, the mask of train set
            val_mask: numpy.ndarray, the mask of validation set
            test_mask: numpy.ndarray, the mask of test set
        '''
        self.dataset = dataset
        self._data = self.load_data_edge_index()
        
    def load_data_edge_index(self):
        current_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(current_path, r'data/')
        data_path = os.path.join(data_path, self.dataset)
        print("data_path:", data_path)

        dataset = Planetoid(root=data_path, name=self.dataset)
        # print("dataset:", dataset)
        
        raw_data_content = pd.read_csv(data_path + '.content', sep='\t', header=None)
        raw_data_cite = pd.read_csv(data_path + '.cites', sep='\t', header=None)

        # extract feature and label
        features = raw_data_content.iloc[:, 1:-1].values
        labels = raw_data_content.iloc[:, -1].values
        
        # convert labels to numbers
        labels_dict = {label: i for i, label in enumerate(np.unique(labels))}
        labels = np.array([labels_dict[label] for label in labels])
        
        # generate edge index
        index_raw = list(raw_data_cite.iloc[:, 0])
        index_new = list(range(len(index_raw)))
        
        
dataload = DataProcessor()