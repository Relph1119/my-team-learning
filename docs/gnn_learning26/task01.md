# Task01 简单图论、环境配置与PyG库

## 1 知识梳理

### 1.1 图深度学习
- 数据类型：矩阵、张量、序列和时间序列
- 图机器学习：构建节点表征，包含节点自身的信息和节点邻接信息
- 挑战：图是不规则的，构造信息复杂

### 1.2 图的表示
- 节点表示实体，边表示实体间的关系
- 类别型的信息为标签，数值型的信息为属性
- 邻接矩阵：表示节点之间边的信息，分为有权和无权，权重用$w$表示

### 1.3 图的属性
- 节点的度：入度$d_{in}(v)$和出度$d_{out}(v)$
- 邻接节点：$N(v_i)$
- 行走：从起点出发，经过边和节点到达终点
- 最短路径：$p_{\mathrm{s} t}^{\mathrm{sp}}=\arg \min _{p \in \mathcal{P}_{\mathrm{st}}}|p|$
- 直径：$\text{diameter}(\mathcal{G})=\max _{v_{s}, v_{t} \in \mathcal{V}} \min _{p \in \mathcal{P}_{s t}}|p|$

### 1.4 应用于图结构数据的机器学习
- 节点预测：预测节点的类别或某类属性的取值
- 边预测：预测两个节点间是否存在链接
- 图预测：对不同的图进行分类或预测图的属性
- 节点聚类：检测节点是否形成一个类

### 1.5 Data类的创建
- x：节点属性矩阵，主要存放节点的相关属性（数据）
- edge_index：边索引矩阵，节点按照x中的编号排序，并通过编号表示节点关系，第1行为尾节点，第2行为头节点，表示头节点指向尾节点
- edge_attr：边属性矩阵，表示边的权重
- y：节点的类型
- Data对象类型转换：`from_dict()`、`to_dict()`

### 1.6 Dataset类
- 创建数据集对象：使用内置数据集类
- 分析数据集样本：通过`train_mask`、`val_mask`、`test_mask`得到训练/验证/测试数据集
- 使用数据集：使用数据集训练模型

## 2 实战练习

**习题：**  
&emsp;&emsp;请通过继承Data类实现一个类，专门用于表示“机构-作者-论文”的网络。该网络包含“机构“、”作者“和”论文”三类节点，以及“作者-机构“和 “作者-论文“两类边。对要实现的类的要求：  
1. 用不同的属性存储不同节点的属性；  
2. 用不同的属性存储不同的边（边没有属性）；  
3. 逐一实现获取不同节点数量的方法。  

**解答思路：**
1. 定义枚举类，用于表示节点的类别：机构-0、作者-1、论文-2
2. 继承Data类，传入参数类型为dataframe
3. 根据Data类初始化，提供x，edge_index，y参数，用于创建Data类
4. 编写create_datasets函数，用于得到Data类初始化用到的参数（x,edge_index,y）
5. 遍历dataframe，按行取出数据：  
  （1）添加节点，由于节点可能有重复，需要进行去重判断  
  （2）构建类型为元组tuple的list，表示(node, label)  
  （3）构建作者与机构、作者与论文的关系，使用np.hstack进行数据合并，得到edge_index  
  （4）从list中得到x, y  
  （5）返回x,edge_index,y
6. 编写测试用例，测试MyDataset是否具备题目要求


```python
from enum import Enum, unique

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@unique
class MyLabel(Enum):
    dept = 0
    paper = 1
    author = 2


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
            node_label_list, dept_index = self.__add_node(
                node_label_list, dept, MyLabel.dept.value)
            node_label_list, paper_index = self.__add_node(
                node_label_list, paper, MyLabel.paper.value)
            node_label_list, author_index = self.__add_node(
                node_label_list, author, MyLabel.author.value)

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
            edge_index = np.hstack(
                (edge_index, author_dept_index)) if edge_index is not None else author_dept_index
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
```


```python
print("有2个作者，分别写了2个论文，来自同一个机构")
# 有2个作者，分别写了2个论文，来自同一个机构
input_data = pd.DataFrame([[1, 1, 1], [1, 2, 2]], columns=[
    'dept', 'paper', 'author'])

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
```

    有2个作者，分别写了2个论文，来自同一个机构
    Number of dept nodes: 1
    Number of author nodes: 2
    Number of paper nodes: 2
    Number of nodes: 5
    Number of edges: 8
    Contains isolated nodes: False
    Contains self-loops: False
    Is undirected: True
    
