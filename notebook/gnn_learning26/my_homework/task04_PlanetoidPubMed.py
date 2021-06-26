#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task04_PlanetoidPubMed.py
@time: 2021/6/23 8:38
@project: my-team-learning
@desc: 
"""

import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, download_url)
from torch_geometric.io import read_planetoid_data


class PlanetoidPubMed(InMemoryDataset):
    r""" 节点代表文章，边代表引文关系。
                 训练、验证和测试的划分通过二进制掩码给出。
    参数:
        root (string): 存储数据集的文件夹的路径
        transform (callable, optional): 数据转换函数，每一次获取数据时被调用。
        pre_transform (callable, optional): 数据转换函数，数据保存到文件前被调用。
    """

    #     url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    url = 'https://gitee.com/jiajiewu/planetoid/raw/master/data'

    def __init__(self, root, transform=None, pre_transform=None):
        super(PlanetoidPubMed, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return ['ind.pubmed.{}'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, 'pubmed')
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':
    dataset = PlanetoidPubMed('dataset/PlanetoidPubMed')
    print(dataset.num_classes)
    print(dataset[0].num_nodes)
    print(dataset[0].num_edges)
    print(dataset[0].num_features)
