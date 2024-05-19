
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
        # self.gcn1 = GraphConvolutioal(input_dim, 512)
        # self.gcn2 = GraphConvolutioal(512, 128)
        # self.gcn3 = GraphConvolutioal(128, 16)
        # self.gcn4 = GraphConvolutioal(16, 7)
        # self.gcn1 = GraphConvolutioal(input_dim, 2048)
        # self.gcn2 = GraphConvolutioal(2048, 1024)
        # self.gcn3 = GraphConvolutioal(1024, 512)
        # self.gcn4 = GraphConvolutioal(512, 256)
        # self.gcn5 = GraphConvolutioal(256, 128)
        # self.gcn6 = GraphConvolutioal(128, 64)
        # self.gcn7 = GraphConvolutioal(64, 16)
        # self.gcn8 = GraphConvolutioal(16, 7)
        # Define parameters
        self.drop_rate = 0.2

        
    def forward(self, features, matrix_sparse):
        '''
        Input:
            features: torch.Tensor, the input features, shape: (num_nodes, input_dim)
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        
        Output:
            h: torch.Tensor, the output features. (classification result)
        '''
        h = nn.functional.relu(self.gcn1(features, matrix_sparse, drop_rate=self.drop_rate))
        # h = nn.functional.relu(self.gcn2(h, matrix_sparse, drop_rate=self.drop_rate))
        # h = nn.functional.relu(self.gcn3(h, matrix_sparse, drop_rate=self.drop_rate))
        # h = nn.functional.relu(self.gcn4(h, matrix_sparse, drop_rate=self.drop_rate))
        # h = nn.functional.relu(self.gcn5(h, matrix_sparse, drop_rate=self.drop_rate))
        # h = nn.functional.relu(self.gcn6(h, matrix_sparse, drop_rate=self.drop_rate))
        # h = nn.functional.relu(self.gcn7(h, matrix_sparse, drop_rate=self.drop_rate))
        h = self.gcn2(h, matrix_sparse, drop_rate=self.drop_rate)
        return h