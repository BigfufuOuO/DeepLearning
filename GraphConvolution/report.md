# GCN
本次实验主要使用GCN
## 设计卷积层
这里根据论文SEMI-SUPERVISED CLASSIFICATION WITH
GRAPH CONVOLUTIONAL NETWORKS中的描述，设计了的卷积层组成如下：首先根据数据中的节点数和边的关系建立邻接矩阵$A$，然后添加自环 $\hat A = A+I$，在此基础上构建度数矩阵 $\hat D$，并令 $ L = \hat D^{-1/2} \hat A \hat D^{-1/2}$. $H^{l+1}$表示第 $l+1$ 层的节点特征矩阵，$$H^{l+1} = \sigma(LH^lW^l)$$
其中 $\sigma$ 表示激活函数，$W^l$ 表示第 $l$ 层的权重矩阵。若需要加一个偏置项，则 $$H^{l+1} = \sigma(LH^lW^l + b^l)$$
## 节点分类
### 加载数据
将特征值和标签存入`features`和`labels`中，其中labels映射到[0,6]的整数上。将节点映射到[0, num_nodes]的整数上，构建邻接矩阵，并将其转化为稀疏矩阵。
```py
import scipy.sparse as sp
for i in range(raw_data_cite.shape[0]):
    index1 = Map_index[raw_data_cite.iloc[i, 0]]
    index2 = Map_index[raw_data_cite.iloc[i, 1]]
    Matrix_adjacency[index1, index2] = 1
    Matrix_adjacency[index2, index1] = 1

Matrix_adjacency = sp.coo_matrix(Matrix_adjacency)
```
随后在稀疏矩阵上加入自环，构建度数矩阵，使用Renomalized trick，得到$L=\hat D^{-1/2} \hat A \hat D^{-1/2}$。
```py
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
        result = result + sp.eye(self.num_codes) # L=L+I
        return result.tocoo()
```
在此基础上，使用固定的随机种子，然后划分训练集、验证集和测试集，比例为4:3:3，最后将数据转化为torch.Tensor。
### GCN设计
在自定义的卷积层设计中，根据$H^{l+1} = \sigma(LH^lW^l + b^l)$，使用稀疏矩阵乘法`torch.sparse.mm`。
```py
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
        features = self.Pair_Norm(features) # PairNorm
        output = torch.mm(features, self.weight) # H * W
        output = torch.sparse.mm(matrix_sparse, output) # A * H * W
        if self.bias is not None:
            output = output + self.bias
        return output
```
如果需要DropEdge，则在稀疏上进行DropEdge操作，即在邻接矩阵上根据概率随机删除边。
```py
 def Drop_edge(self, drop_rate, matrix_sparse):
        '''
        Input:
            drop_rate: float, the rate of drop edge
        
        Output:
            matrix_sparse: torch.Tensor, the sparse matrix of adjacency matrix, shape: (num_nodes, num_nodes)
        '''
        matrix_sparse = matrix_sparse.coalesce() #进一步压缩稀疏矩阵
        num_edges = matrix_sparse._nnz()
        edge_index = matrix_sparse._indices()
        edge_value = matrix_sparse._values()
        drop_num = int(num_edges * drop_rate)
        drop_index = torch.randint(0, num_edges, (drop_num,))
        edge_value[drop_index] = 0
        matrix_sparse = torch.sparse_coo_tensor(edge_index, edge_value, matrix_sparse.shape)
        return matrix_sparse
```
若需要加入PairNorm，则在输入特征上进行归一化，即$\hat H = PairNorm(H)$后再输入到GCN中。其中根据论文Pairnorm: Tackling oversmoothing in GNNs中的描述，PairNorm需要做两步操作：第一步为将所有特征点中心化，即将重心设置为原点
$$ \hat H = H - \frac{1}{n} \sum_{i=1}^{n} H_i$$
第二步是将所有特征点归一化，即将所有特征点的范数归一化为$s$（默认为1）
$$ \hat H_i = s\frac{\hat H_i}{\|\hat H_i \|}$$
### 包装
在GCNNet中包装自定义设计的GCN层，定义输入维度和输出维度，以及Dropout的概率。激活函数默认使用ReLU。
```py
def __init__(self, input_dim):
        '''
        Input:
            input_dim: int, the dimension of input features
        '''
        super(GCNNet, self).__init__()
        self.gcn1 = GraphConvolutioal(input_dim, 16)
        self.gcn2 = GraphConvolutioal(16, 7)
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
        h = self.gcn2(h, matrix_sparse, drop_rate=self.drop_rate)
        return h
```
### 自环
自环的目的是将自身节点的特征加入计算中，增强节点自身的特征表示。节点在更新其特征表示时也会考虑自身的原始特征，从而保留更多自身的信息。在本实验中，自环的添加是在上加上单位矩阵。自环的影响对于GCN的表达能力有显著的影响，以下是训练后再测试集上的正确率，其中`drop_rate=0.2`，不开启PairNorm和偏置。
层数 |  自环(在$L$上) | 无自环(在$A$上) |自环(在$A$上) |
-|-|-|-
 2 | 0.802 | 0.763 | **0.834**
 4 | 0.810 | 0.767 | **0.821**
 8 | 0.740 | 0.742 | **0.822**

其中，自环(在$L$上)指 $L=I +(D+I)^{-1/2} (A+I)  (D+I)^{-1/2}$，无自环(在$A$上)指 $L=D^{-1/2}A  D^{-1/2}$，自环(在$A$上)指$L=(D+I)^{-1/2} (A+I)  (D+I)^{-1/2}$
在令$L=I +(D+I)^{-1/2} (A+I)  (D+I)^{-1/2}$时，出现了比较严重的过拟合现象。此外，在完全不加自环时，出现了很严重的振荡，即使此时使用早停模型的准确率也无法提高。![alt text](classified/figs/cora_loss_acc_0519-1840.png)
### Dropedge
DropEdge即随机丢弃一些边，该操作在L上进行实现，即令 $L \leftarrow DropEdge(L)$，使用DropEdge可以一定程度减少过拟合现象，并且

## 参考资料
https://arxiv.org/pdf/1609.02907
https://arxiv.org/pdf/1907.10903
https://arxiv.org/abs/1909.12223
http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML2020/GNN.pdf
https://blog.csdn.net/qq_38463737/article/details/109636779
https://ifwind.github.io/2021/06/25/图神经网络的下游任务2-链路预测/#加载数据

