
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ConvolutionalLayer import GraphConvolutioal

class GCNNet(nn.Module):
    def __init__(self, input_dim, drop_rate=0.2):
        '''
        Input:
            input_dim: int, the dimension of input features
        '''
        super(GCNNet, self).__init__()
        # Define parameters
        self.drop_rate = drop_rate
        self.PairNorm = False
        self.bias = False
        self.gcn1 = GraphConvolutioal(input_dim, 16, drop_rate=self.drop_rate, PairNorm=self.PairNorm, bias=self.bias)
        self.gcn2 = GraphConvolutioal(16, 7, drop_rate=self.drop_rate, PairNorm=self.PairNorm, bias=self.bias)
        

        
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