
import torch.nn as nn
from ConvolutionalLayer import GraphConvolutioal

class GCNNet(nn.Module):
    def __init__(self, input_dim):
        '''
        Input:
            input_dim: int, the dimension of input features
        '''
        super(GCNNet, self).__init__()
        self.gcn1 = GraphConvolutioal(input_dim, 16)
        self.gcn2 = GraphConvolutioal(16, 7)
        
    def forward(self, features, matrix_sparse):
        '''
        Input:
            features: torch.Tensor, the input features, shape: (num_nodes, input_dim)
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        
        Output:
            h: torch.Tensor, the output features. (classification result)
        '''
        h = nn.functional.relu(self.gcn1(features, matrix_sparse))
        h = self.gcn2(h, matrix_sparse)
        return h