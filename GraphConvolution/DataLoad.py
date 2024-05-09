# Get data from dataset
import os
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, dataset='cora'):
        self.dataset = dataset

    def load_data_adjacency(self):
        current_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(current_path, 'data/')
        data_path = os.path.join(data_path, self.dataset + '/' + self.dataset)

        raw_data_content = pd.read_csv(data_path + '.content', sep='\t', header=None)
        raw_data_cite = pd.read_csv(data_path + '.cites', sep='\t', header=None)

        # extract feature and label
        features = raw_data_content.iloc[:, 1:-1].values
        labels = raw_data_content.iloc[:, -1].values

        # construct adjacency matrix
        ## Construct zero matrix
        print(raw_data_content.shape)
        num_codes = raw_data_content.shape[0] # 2708
        Matrix_adjacency = np.zeros((num_codes, num_codes))

        ## transform index
        index_raw = list(raw_data_content.iloc[:, 0])
        index_new = list(range(num_codes))
        Map_index = dict(zip(index_raw, index_new))

        ## construct adjacency matrix
        for i in range(raw_data_cite.shape[0]):
            index1 = Map_index[raw_data_cite.iloc[i, 0]]
            index2 = Map_index[raw_data_cite.iloc[i, 1]]
            Matrix_adjacency[index1, index2] = 1
            Matrix_adjacency[index2, index1] = 1

        # construct degree matrix
        # Matrix_degree = np.diag(np.sum(Matrix_adjacency, axis=1))
        
        
    def load_data_sparse(self):
        current_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(current_path, 'data/')
        data_path = os.path.join(data_path, self.dataset + '/' + self.dataset)

        raw_data_content = pd.read_csv(data_path + '.content', sep='\t', header=None)
        raw_data_cite = pd.read_csv(data_path + '.cites', sep='\t', header=None)

        # extract feature and label
        features = raw_data_content.iloc[:, 1:-1].values
        labels = raw_data_content.iloc[:, -1].values
        
        # Construct sparse matrix
        ## transform index
        num_codes = raw_data_content.shape[0] # 2708
        index_raw = list(raw_data_content.iloc[:, 0])
        index_new = list(range(num_codes))
        Map_index = dict(zip(index_raw, index_new))
        
        ## construct sparse matrix
        
cora = DataProcessor()
cora.load_data('cora')

