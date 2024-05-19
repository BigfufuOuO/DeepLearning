# Get data from dataset
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data
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
        self._data = self.load_data_adjacency()

    def load_data_adjacency(self):
        current_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(current_path, 'data/')
        data_path = os.path.join(data_path, self.dataset + '/' + self.dataset)

        raw_data_content = pd.read_csv(data_path + '.content', sep='\t', header=None, low_memory=False)
        raw_data_cite = pd.read_csv(data_path + '.cites', sep='\t', header=None, low_memory=False)

        # extract feature and label
        features = raw_data_content.iloc[:, 1:-1].values
        labels = raw_data_content.iloc[:, -1].values
        print("labels:", labels.shape)
        
        # convert labels to numbers
        labels_dict = {label: i for i, label in enumerate(np.unique(labels))}
        labels = np.array([labels_dict[label] for label in labels])

        # construct adjacency matrix
        ## Construct zero matrix
        print(raw_data_content.shape)
        num_codes = raw_data_content.shape[0] # 2708
        print("data size: ", num_codes)
        self.num_codes = num_codes
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
        Matrix_degree = np.diag(np.sum(Matrix_adjacency, axis=1))
        
        # construct sparse matrix 
        Matrix_sparse_adj = self.construct_sparse_matrix(Matrix_adjacency)
        
        # Seperate dataset
        train_mask, val_mask, test_mask = self.Seperate_dataset()
        return Data(Matrix_sparse_adj, Matrix_degree, features, labels, train_mask, val_mask, test_mask)
        
    def Seperate_dataset(self):
        # seperate to train, validation adn test set
        num_train = (int) (self.num_codes * 0.4)
        num_val = (int) (self.num_codes * 0.3)
        num_test = (int) (self.num_codes - num_train - num_val)
        
        # set a random seed
        np.random.seed(42)
        index = np.random.permutation(self.num_codes)
        index_train = index[:num_train]
        index_val = index[num_train:num_train + num_val]
        index_test = index[num_train + num_val:]
        
        # generate vector
        train_mask = np.zeros(self.num_codes, dtype=bool)
        val_mask = np.zeros(self.num_codes, dtype=bool)
        test_mask = np.zeros(self.num_codes, dtype=bool)
        train_mask[index_train] = True
        val_mask[index_val] = True
        test_mask[index_test] = True
        
        return train_mask, val_mask, test_mask
        
    def construct_sparse_matrix(self, matrix):
        return sp.coo_matrix(matrix)
    
    def Renormalization_trick(self):
        '''
        Caculate L = I+D^(-1/2) (A + I) D^(-1/2)
        Output:
            L: scipy.sparse.coo_matrix, the renormalized matrix
        '''
        Matrix = self._data.Matrix_sparse + sp.eye(self.num_codes) # A + I
        degree = np.array(Matrix.sum(1))
        degree = np.power(degree, -0.5)
        degree_hat = sp.diags(degree.flatten())
        result = degree_hat.dot(Matrix).dot(degree_hat)
        result = result + sp.eye(self.num_codes)
        return result.tocoo()
    
    def Transform_Data(self):
        Normalized_feature = self._data.features / self._data.features.sum(1, keepdims=True)
        tensor_feature = torch.from_numpy(Normalized_feature).to(device)
        tensor_feature = tensor_feature.to(torch.float32)
        tensor_labels = torch.from_numpy(self._data.labels).to(device)
        print("tensor_labels:", tensor_labels.shape)
        
        tensor_train_mask = torch.from_numpy(self._data.train_mask).to(device)
        tensor_val_mask = torch.from_numpy(self._data.val_mask).to(device)
        tensor_test_mask = torch.from_numpy(self._data.test_mask).to(device)
        
        #Normalized_matrix_sparse = self.Renormalization_trick()
        Normalized_matrix_sparse = self._data.Matrix_sparse
        
        # construct sparse matrix tensor
        indices = torch.from_numpy(np.vstack((Normalized_matrix_sparse.row, Normalized_matrix_sparse.col)).astype(np.int64))
        values = torch.from_numpy(Normalized_matrix_sparse.data.astype(np.float32))
        tensor_matrix_sparse = torch.sparse_coo_tensor(indices, values, Normalized_matrix_sparse.shape).to(device)
        
        return Data(tensor_matrix_sparse, self._data.Matrix_degree, tensor_feature, tensor_labels, \
                    tensor_train_mask, tensor_val_mask, tensor_test_mask)
        
        

# cora = DataProcessor('cora')
# cora.Transform_Data()
pass
