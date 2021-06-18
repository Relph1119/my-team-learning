#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task02.py
@time: 2021/6/19 5:49
@project: my-team-learning
@desc: 
"""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


if __name__ == '__main__':
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