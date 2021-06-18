# Task02 消息传递范式

## 1 知识梳理

### 1.1 消息传递范式介绍
- 概念：一种聚合邻接节点信息来更新中心节点信息的范式
- 过程：通过邻接节点信息经过变换后聚合，在所有节点上进行一遍，多次更新后的节点信息就作为**节点表征**
- 消息传递图神经网络：$$ \mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right)$$
- 节点嵌入：神经网络生成节点表征的操作

### 1.2 `MessagePassing`基类
- 作用：封装了消息传递的运行流程
- `aggr`：聚合方案，`flow`：消息传递的流向，`node_dim`：传播的具体维度
- `MessagePassing.propagate()`：开始传递消息的起始调用
- `MessagePassing.message()`：实现$\phi$函数
- `MessagePassing.aggregate()`：从源节点传递过来的消息聚合在目标节点上的函数，使用`sum`,`mean`和`max`
- `MessagePassing.update()`：实现$\gamma$函数

### 1.3 GCNConv示例
- 数学公式：$$\mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot \left( \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)} \right)$$
- 矩阵形式：$$\mathbf{X}' = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}\mathbf{X}\Theta$$
- 步骤：
  1. 向邻接矩阵添加自循环边：构建$\mathbf{\hat{A}}$
  2. 对节点表征进行线性变换：计算$\mathbf{X}\Theta$
  3. 计算归一化系数：计算$\mathbf{\hat{D}}^{-1/2}$
  4. 将相邻节点表征相加（`add`聚合）：得到一个对称且归一化的矩阵


```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # 使用add聚合函数
        # 根据公式j<-N(i)，表示消息从源节点`i`传播到目标节点`j`，flow='source_to_target'
        # 定义线性变换
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x的维度是[N, in_channels]
        # 邻接矩阵的维度是[2, E]

        # Step 1: 添加自循环边，构建A
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: 对节点进行线性变换
        x = self.lin(x)

        # Step 3: 计算归一化系数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: 调用propagate函数，开启消息传递
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j的维度是[E, out_channels]
        # Step 4: 将x_j进行归一化
        return norm.view(-1, 1) * x_j
```


```python
# 随机种子
torch.manual_seed(0)

# 定义边
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# 定义节点特征，每个节点特征维度是2
x = torch.tensor([[-1, 2], [0, 4], [1, 5]], dtype=torch.float)

# 创建一层GCN层，并把特征维度从2维降到1维
conv = GCNConv(2, 1)

# 前向传播
x = conv(x, edge_index)
print(x)
print(conv.lin.weight)
```

    tensor([[0.4728],
            [0.9206],
            [1.0365]], grad_fn=<ScatterAddBackward>)
    Parameter containing:
    tensor([[-0.0053,  0.3793]], requires_grad=True)
    

## 2 实战练习

### 2.1 习题1 
请总结`MessagePassing`基类的运行流程

**解答：**

`MessagePassing`基类的运行流程：
1. 初始化参数聚合函数`aggr`，消息传递流向`flow`，传播维度`node_dim`
2. 初始化自实现函数中用到的自定义参数`__user_args__`，`__fused_user_args__`
3. 基于`Module`基类，调用`forward`函数，用于数据或参数的初始化
4. `propagate`函数：  
  （1）检查`edge_index`和`size`参数是否符合要求，并返回`size`  
  （2）判断`edge_index`是否为`SparseTensor`，如果满足，则执行`message_and_aggregate`，再执行`update`方法  
  （3）如果不满足，就先执行`message`方法，再执行`aggregate`和`update`方法  

### 2.2 习题2
请复现一个一层的图神经网络的构造，总结通过继承`MessagePassing`基类来构造自己的图神经网络类的规范。

## 3 参考文章

【1】理解GCN的整个算法流程：https://blog.csdn.net/qq_41987033/article/details/103377561
