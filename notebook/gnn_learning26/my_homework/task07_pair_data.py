#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task07_pair_data.py
@time: 2021/7/5 20:16
@project: my-team-learning
@desc: 图的匹配
"""
import torch
from torch_geometric.data import Data, DataLoader


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


if __name__ == '__main__':
    edge_index_s = torch.tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    x_s = torch.randn(5, 16)  # 5 nodes.
    edge_index_t = torch.tensor([
        [0, 0, 0],
        [1, 2, 3],
    ])
    x_t = torch.randn(4, 16)  # 4 nodes.

    data = PairData(edge_index_s, x_s, edge_index_t, x_t)
    data_list = [data, data]
    loader = DataLoader(data_list, batch_size=2)
    batch = next(iter(loader))

    print(batch)
    print(batch.edge_index_s)
    print(batch.edge_index_t)
