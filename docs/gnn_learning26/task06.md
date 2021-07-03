# Task06 基于图神经网络的图表征学习方法

## 1 知识梳理

### 1.1 基于图同构网络(GIN)的图表征网络的实现
- 基本思路：计算节点表征，对图上各个节点的表征做图池化，得到图的表征
- 图表征模块(GINGraphReprModule)：
  1. 对图上每一个节点进行节点嵌入，得到节点表征
  2. 对节点表征做图池化，得到图表征
  3. 用一层线性变换对图表征转换为对图的预测
- 节点嵌入模块(GINNodeEmbeddingModule)：
  1. 用AtomEcoder进行嵌入，得到第0层节点表征
  2. 逐层计算节点表征
  3. 感受野越大，节点i的表征最后能捕获到节点i的距离为num_layers的邻接节点的信息
- 图同构卷积层(GINConv)：
  1. 数学定义：$ \displaystyle \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)$
  2. 将类别型边属性转换为边表征
  3. 消息传递、消息聚合、消息更新
- AtomEncoder和BondEncoder：将节点属性和边属性分布映射到一个新空间，再对节点和边进行信息融合。

### 1.2 WL Test
- 图同构性测试：
  1. 迭代聚合节点及其邻接节点的标签
  2. 将聚合标签散列成新的标签，数学公式：$\displaystyle L^{h}_{u} \leftarrow \operatorname{hash}\left(L^{h-1}_{u} + \sum_{v \in \mathcal{N}(U)} L^{h-1}_{v}\right)$
- WL子树核衡量图之间相似性：使用不同迭代中的节点标签计数作为图的表征向量
- 详细步骤：
  1. 聚合自身与邻接节点的标签，得到一串字符串
  2. 标签散列，将较长的字符串映射到一个简短的标签
  3. 给节点重新打上标签
- 图相似性评估：
  1. WL Subtree Kernel方法：用WL Test算法得到节点多层标签，统计图中各类标签出现的次数，使用向量表示，作为图的表征
  2. 两个图的表征向量内积，作为两个图的相似性估计
- 判断图同构性的必要条件：两个节点自身标签一样且它们的邻接节点一样，将两个节点映射到相同的表征

## 2 实战练习


```python
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from ogb.graphproppred.mol_encoder import BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
```

### 2.1 GIN的图表征模块


```python
class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="sum"):
        """GIN Graph Pooling Module

        此模块首先采用GINNodeEmbedding模块对图上每一个节点做嵌入，然后对节点嵌入做池化得到图的嵌入，最后用一层线性变换得到图的最终的表示（graph representation）。

        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表示的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        # 对图上的每个节点进行节点嵌入
        self.gnn_node = GINNodeEmbedding(
            num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        # 一层线性变换，对图表征转换为对图的预测
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)
```

### 2.2 节点嵌入模块


```python
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module
        采用多层GINConv实现图上结点的嵌入。
        """

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.atom_encoder = AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        # computing input node embedding
        # 先将类别型原子属性转化为原子嵌入，得到第0层节点表征
        h_list = [self.atom_encoder(x)]  
        # 逐层计算节点表征
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]
            
            # 得到全部节点表征
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation
```

### 2.3 图同构卷积层


```python
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(
            emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)  # 先将类别型边属性转换为边嵌入
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
```

### 2.4 AtomEncoder与BondEncoder


```python
full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(torch.nn.Module):
    """该类用于对原子属性做嵌入。
    记`N`为原子属性的维度，则原子属性表示为`[x1, x2, ..., xi, xN]`，其中任意的一维度`xi`都是类别型数据。full_atom_feature_dims[i]存储了原子属性`xi`的类别数量。
    该类将任意的原子属性`[x1, x2, ..., xi, xN]`转换为原子的嵌入`x_embedding`（维度为emb_dim）。
    """

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)  # 不同维度的属性用不同的Embedding方法
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        # 节点的不同属性融合
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
         # 边的不同属性融合
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
```
