#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task02_myGNN.py
@time: 2021/6/19 7:29
@project: my-team-learning
@desc: 自实现一层图神经网络
"""
import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid


class MyGNN(MessagePassing):
    """
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 -
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i + \mathbf{\Theta}_3 \mathbf{x}_j)
    """

    def __init__(self, in_channels, out_channels, device):
        super(MyGNN, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = torch.nn.Linear(in_channels, out_channels).to(device)
        self.lin2 = torch.nn.Linear(in_channels, out_channels).to(device)
        self.lin3 = torch.nn.Linear(in_channels, out_channels).to(device)

    def forward(self, x, edge_index):
        a = self.lin1(x)
        b = self.lin2(x)
        out = self.propagate(edge_index, a=a, b=b)
        return self.lin3(x) - out

    def message(self, a_i, b_j):
        out = a_i + b_j
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


if __name__ == '__main__':
    device = torch.device('cuda:0')

    dataset = Planetoid(root='dataset/Cora', name='Cora')
    model = MyGNN(in_channels=dataset.num_features, out_channels=dataset.num_classes, device=device)
    print(model)

    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).to(device)
        pred = out.argmax(dim=1)
        accuracy = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / data.test_mask.sum()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("Train Epoch: {} Accuracy: {:.2f}%".format(epoch, accuracy.item() * 100.0))
