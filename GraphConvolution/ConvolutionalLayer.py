import torch
import torch.nn as nn

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
    
    def forward(self, features, matrix_sparse):
        '''
        Input:
            features: torch.Tensor, the input features, shape: (num_nodes, input_dims)
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        
        Output:
            output: torch.Tensor, the output features, shape: (num_nodes, output_dims)
            
        Note:
            H(l+1) = A * H(l) * W(l) + b(l), where A is could be renormalized.
        '''
        output = torch.mm(features, self.weight) # H * W
        output = torch.sparse.mm(matrix_sparse, output) # A * H * W
        if self.bias is not None:
            output = output + self.bias
        return output