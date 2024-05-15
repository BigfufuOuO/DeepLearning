import torch.nn as nn
import torch
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
        self.gcn1 = GraphConvolutioal(input_dim, 128)
        self.gcn2 = GraphConvolutioal(128, 64)
        # Define parameters
        self.drop_rate = 0.2

        
    def encoder(self, features, matrix_sparse):
        '''
        Input:
            features: torch.Tensor, the input features, shape: (num_nodes, input_dim)
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        
        Output:
            h: torch.Tensor, the output features. (classification result)
        '''
        h = nn.functional.relu(self.gcn1(features, matrix_sparse, drop_rate=self.drop_rate))
        h = self.gcn2(h, matrix_sparse, drop_rate=self.drop_rate)
        return h
    
    def decoder(self, h, positive_edge_pairs, negative_edge_pairs):
        '''
            positive_edge_pairs: torch.Tensor, the positive edge pairs, shape: (2, num_positive_edges)
            negative_edge_pairs: torch.Tensor, the negative edge pairs, shape: (2, num_negative_edges)
        '''
        all_pairs = torch.cat((positive_edge_pairs, negative_edge_pairs), dim=-1)
        result = (h[all_pairs[0]] * h[all_pairs[1]]).sum(dim=-1)
        return result
        pass
    
    def forward(self, feaatures, matrix_sparse, positive_edge_pairs, negative_edge_pairs):
        result = self.decode(self.encode(feaatures, matrix_sparse), positive_edge_pairs, negative_edge_pairs)
        return result