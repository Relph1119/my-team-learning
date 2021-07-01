# Task05 超大图上的节点表征学习

## 1 知识梳理

### 1.1 Cluster-GCN方法
- 基本思路：利用图节点聚类算法，将一个图的节点划分为多个簇，每一次选择几个簇的节点和这些节点对应边，构成一个子图，然后对子图训练
- 效果：
    1. 提高表征利用率，提高图神经网络的训练效率
    2. 随机选择多个簇组成batch，保证batch内类别分布
    3. 内存空间的优化

### 1.2 节点表征学习
- 节点表征递推公式：$\displaystyle Z^{(l+1)}=A^{\prime} X^{(l)} W^{(l)}, X^{(l+1)}=\sigma\left(Z^{(l+1)}\right)$
- 训练目标：最小化损失函数来学习权重矩阵$\displaystyle \mathcal{L}=\frac{1}{\left|\mathcal{Y}_{L}\right|} \sum_{i \in \mathcal{Y}_{L}} \operatorname{loss}\left(y_{i}, z_{i}^{L}\right)$

### 1.3 Cluster-GCN方法详细分析
- 内存消耗：存储所有的节点表征矩阵，需要$O(NFL)$空间
- mini-batch SGD方式训练：不需要计算完整梯度，只需要计算部分梯度
- 训练时间：一个节点梯度计算需要$O(d^L F^2)$时间
- 节点表征的利用率：在训练过程中，如果节点$i$在$l$层的表征$z_{i}^{(l)}$被计算并在$l+1$层的表征计算中被重复使用$u$次

### 1.4 简单的Cluster-GCN方法
- 基本步骤：
  1. 将节点划分为$c$个簇，邻接矩阵被划分为大小为$c^2$的块矩阵
  2. 用块对角线邻接矩阵$\bar{A}$去近似邻接矩阵$A$
  3. 采样一个簇$\mathcal{V}_{t}$，根据$\mathcal{L}_{{\bar{A}^{\prime}}_{tt}}$的梯度进行参数更新
- 时间复杂度：
  1. 每个batch的总体时间复杂度为$O\left(\left\|A_{t t}\right\|_{0} F+ b F^{2}\right)$
  2. 每个epoch的总体时间复杂度为$O\left(\|A\|_{0} F+N F^{2}\right)$
  3. 总时间复杂度为$O\left(L\|A\|_{0} F+LN F^{2}\right)$
- 表征数：每个batch只需要计算$O(b L)$的表征
- 空间复杂度：用于存储表征的内存为$O(bLF)$，总空间复杂度为$O(bLF + LF^2)$

## 2 实战练习


```python
from torch_geometric.datasets import Reddit

dataset = Reddit('dataset/Reddit')
data = dataset[0]
print("类别数：", dataset.num_classes)
print("节点数：", data.num_nodes)
print("边数：", data.num_edges)
print("节点维度：", data.num_features)
```

    类别数： 41
    节点数： 232965
    边数： 114615892
    节点维度： 602
    


```python
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

# 图节点聚类
cluster_data = ClusterData(
    data, num_parts=1500, recursive=False, save_dir=dataset.processed_dir)
train_loader = ClusterLoader(
    cluster_data, batch_size=20, shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(
    data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=12)
```


```python
import torch
from torch.nn import ModuleList
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.convs = ModuleList(
            [SAGEConv(in_channels, 128),
             SAGEConv(128, out_channels)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
```


```python
def train():
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)
    y_pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(1, 31):
    loss = train()
    if epoch % 5 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, test: {test_acc:.4f}')
    else:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
```

    Epoch: 01, Loss: 1.1573
    Epoch: 02, Loss: 0.4642
    Epoch: 03, Loss: 0.3904
    Epoch: 04, Loss: 0.3572
    

    Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 465930/465930 [01:05<00:00, 7123.90it/s]
    

    Epoch: 05, Loss: 0.3385, Train: 0.9579, Val: 0.9535, test: 0.9520
    Epoch: 06, Loss: 0.3139
    Epoch: 07, Loss: 0.3060
    Epoch: 08, Loss: 0.2950
    Epoch: 09, Loss: 0.2933
    

    Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 465930/465930 [01:02<00:00, 7484.08it/s]
    

    Epoch: 10, Loss: 0.2879, Train: 0.9609, Val: 0.9528, test: 0.9509
    Epoch: 11, Loss: 0.2902
    Epoch: 12, Loss: 0.2871
    Epoch: 13, Loss: 0.2803
    Epoch: 14, Loss: 0.3023
    

    Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 465930/465930 [01:01<00:00, 7548.35it/s]
    

    Epoch: 15, Loss: 0.2796, Train: 0.9571, Val: 0.9465, test: 0.9436
    Epoch: 16, Loss: 0.2711
    Epoch: 17, Loss: 0.2619
    Epoch: 18, Loss: 0.2726
    Epoch: 19, Loss: 0.2710
    

    Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 465930/465930 [01:02<00:00, 7476.75it/s]
    

    Epoch: 20, Loss: 0.2538, Train: 0.9663, Val: 0.9501, test: 0.9499
    Epoch: 21, Loss: 0.2532
    Epoch: 22, Loss: 0.2457
    Epoch: 23, Loss: 0.2470
    Epoch: 24, Loss: 0.2340
    

    Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 465930/465930 [01:03<00:00, 7384.29it/s]
    

    Epoch: 25, Loss: 0.2428, Train: 0.9666, Val: 0.9503, test: 0.9491
    Epoch: 26, Loss: 0.2469
    Epoch: 27, Loss: 0.2340
    Epoch: 28, Loss: 0.2511
    Epoch: 29, Loss: 0.2444
    

    Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 465930/465930 [01:02<00:00, 7445.58it/s]

    Epoch: 30, Loss: 0.2278, Train: 0.9684, Val: 0.9516, test: 0.9507
    

    
    
