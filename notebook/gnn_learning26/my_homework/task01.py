#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task01.py
@time: 2021/6/16 0:23
@project: my-team-learning
@desc:
请通过继承Data类实现一个类，专门用于表示“机构-作者-论文”的网络。该网络包含“机构“、”作者“和”论文”三类节点，以及“作者-机构“和 “作者-论文“两类边。
对要实现的类的要求：
1）用不同的属性存储不同节点的属性；
2）用不同的属性存储不同的边（边没有属性）；
3）逐一实现获取不同节点数量的方法。
"""

from torch_geometric.data import Data


class MyData(Data):
    def __init__(self, dept_x, author_x, paper_x, author_dept_edge_index, author_paper_edge_index, y, **kwargs):
        self.dept_x = dept_x
        self.author_x = author_x
        self.paper_x = paper_x
        self.author_dept_edge_index = author_dept_edge_index
        self.author_paper_edge_index = author_paper_edge_index
        self.y = y
        super().__init__(**kwargs)

    @property
    def dept_nums(self):
        return self.dept_x.shape[0]

    @property
    def author_nums(self):
        return self.author_x.shape[0]

    @property
    def paper_nums(self):
        return self.paper_x.shape[0]

    @property
    def dept_feats(self):
        return self.dept_x.shape[1]

    @property
    def author_feats(self):
        return self.author_x.shape[1]

    @property
    def paper_feats(self):
        return self.paper_x.shape[1]


