import torch
import torch.nn as nn
from torch_geometric.nn import PairNorm

class GraphConvolutioal(nn.Module):
    def __init__(self, input_dims, output_dims, bias=False):
        '''
        Input:
            input_dims: int, the dimension of input features
            output_dims: int, the dimension of output features
            bias: bool, whether to use bias
        
        Member:
            input_dims: int, the dimension of input features
            output_dims: int, the dimension of output features
            weight: torch.nn.Parameter, the weight matrix
            bias: torch.nn.Parameter, the bias vector
        '''
        super(GraphConvolutioal, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weight = nn.Parameter(torch.FloatTensor(input_dims, output_dims))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dims))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, features, matrix_sparse, drop_rate = 0.2):
        '''
        Input:
            features: torch.Tensor, the input features, shape: (num_nodes, input_dims)
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        
        Output:
            output: torch.Tensor, the output features, shape: (num_nodes, output_dims)
            
        Note:
            H(l+1) = A * H(l) * W(l) + b(l), where A is could be renormalized.
        '''
        matrix_sparse = self.Drop_edge(drop_rate, matrix_sparse) # DropEdge
        # features = self.Pair_Norm(features) # PairNorm
        output = torch.mm(features, self.weight) # H * W
        output = torch.sparse.mm(matrix_sparse, output) # A * H * W
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def Drop_edge(self, drop_rate, matrix_sparse):
        '''
        Input:
            drop_rate: float, the rate of drop edge
        
        Output:
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        '''
        matrix_sparse = matrix_sparse.coalesce()
        num_edges = matrix_sparse._nnz()
        edge_index = matrix_sparse._indices()
        edge_value = matrix_sparse._values()
        drop_num = int(num_edges * drop_rate)
        drop_index = torch.randint(0, num_edges, (drop_num,))
        edge_value[drop_index] = 0
        matrix_sparse = torch.sparse_coo_tensor(edge_index, edge_value, matrix_sparse.shape)
        return matrix_sparse

    def Pair_Norm(self, features):
        '''
        Pair normalization.
        Input:
            features: torch.Tensor, the input features, shape: (num_nodes, input_dims)
        Output:
            Normalized features.
        '''
        features = PairNorm(scale_individually=True)(features)
        return features
        
        
