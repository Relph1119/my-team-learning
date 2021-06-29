#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task01_mydata.py
@time: 2021/6/15 14:07
@project: my-team-learning
@desc:
请通过继承Data类实现一个类，专门用于表示“机构-作者-论文”的网络。该网络包含“机构“、”作者“和”论文”三类节点，以及“作者-机构“和 “作者-论文“两类边。
对要实现的类的要求：
1）用不同的属性存储不同节点的属性；
2）用不同的属性存储不同的边（边没有属性）；
3）逐一实现获取不同节点数量的方法。
"""

from enum import Enum, unique

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


@unique
class MyLabel(Enum):
    dept = 0
    paper = 1
    author = 2

    @staticmethod
    def get_name(val):
        return MyLabel(val).name


class MyDataset(Data):
    def __init__(self, input_data, **kwargs):
        self.data = input_data
        # 构建数据集需要的参数
        x, edge_index, y, author_dept_edge_index, author_paper_edge_index = self.__create_datasets()
        self.author_dept_edge_index = author_paper_edge_index
        self.author_paper_edge_index = author_paper_edge_index
        super().__init__(x=x, edge_index=edge_index, y=y, **kwargs)

    def __add_node(self, node_label_list, node, label):
        if node_label_list.count((node, label)) == 0:
            # 添加节点
            node_label_list.append((node, label))

        node_index = node_label_list.index((node, label))

        # 返回节点集，节点索引
        return node_label_list, node_index

    def __create_datasets(self):
        node_label_list = []
        edge_index = None
        author_dept_edge_index = None
        author_paper_edge_index = None

        for row in self.data.values.tolist():
            # 取出三个节点数据
            dept = row[0]
            paper = row[1]
            author = row[2]

            # 添加节点，得到节点索引
            node_label_list, dept_index = self.__add_node(node_label_list, dept, MyLabel.dept.value)
            node_label_list, paper_index = self.__add_node(node_label_list, paper, MyLabel.paper.value)
            node_label_list, author_index = self.__add_node(node_label_list, author, MyLabel.author.value)

            # 构建作者与机构的关系
            author_dept_index = np.array([[author_index, dept_index],
                                          [dept_index, author_index]])

            author_dept_edge_index = np.hstack((author_dept_edge_index, author_dept_index)) \
                if author_dept_edge_index is not None else author_dept_index
            # 构建作者与论文的关系
            author_paper_index = np.array([[author_index, paper_index],
                                           [paper_index, author_index]])

            author_paper_edge_index = np.hstack((author_paper_edge_index, author_paper_index)) \
                if author_paper_edge_index is not None else author_paper_index

            # 添加边的索引
            edge_index = np.hstack((edge_index, author_dept_index)) if edge_index is not None else author_dept_index
            edge_index = np.hstack((edge_index, author_paper_index))

        nodes = [[node] for node, label in node_label_list]
        labels = [[label] for node, label in node_label_list]

        x = torch.tensor(nodes)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor(labels)
        return x, edge_index, y, author_dept_edge_index, author_paper_edge_index

    @property
    def dept_nums(self):
        return self.data['dept'].nunique()

    @property
    def author_nums(self):
        return self.data['author'].nunique()

    @property
    def paper_nums(self):
        return self.data['paper'].nunique()


if __name__ == '__main__':
    print("有2个作者，分别写了2个论文，来自同一个机构")
    # 有2个作者，分别写了2个论文，来自同一个机构
    input_data = pd.DataFrame([[1, 1, 1], [2, 2, 2]], columns=['dept', 'paper', 'author'])

    data = MyDataset(input_data)
    print("Number of dept nodes:", data.dept_nums)
    print("Number of author nodes:", data.author_nums)
    print("Number of paper nodes:", data.paper_nums)
    # 节点数量
    print("Number of nodes:", data.num_nodes)
    # 边数量
    print("Number of edges:", data.num_edges)
    # 此图是否包含孤立的节点
    print("Contains isolated nodes:", data.contains_isolated_nodes())
    # 此图是否包含自环的边
    print("Contains self-loops:", data.contains_self_loops())
    # 此图是否是无向图
    print("Is undirected:", data.is_undirected())

    plt.figure(figsize=(6, 6))
    G = nx.Graph()
    index = 0
    x = data.x.tolist()
    y = data.y.tolist()
    for x_name, y_label in zip(x, y):
        G.add_node(index, label=MyLabel.get_name(y_label[0]) + '-' + str(x_name[0]))
        index += 1

    edge_index = [(i, j) for i, j in zip(data.edge_index.tolist()[0], data.edge_index.tolist()[1])]
    G.add_edges_from(edge_index)
    pos = nx.spring_layout(G, iterations=20)
    nx.draw(G, pos, edge_color="grey", node_size=500)  # 画图，设置节点大小
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=10)
    plt.show()
