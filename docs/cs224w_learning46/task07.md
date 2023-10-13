# Task07 标签传播与节点分类

## 1 基本概念回顾

- 图机器学习的任务：节点层面、连接层面、社交（子图）层面、全图层面
- 半监督节点分类：通过已知节点预测出未知标签节点的类别
- 半监督节点分类问题的求解方法：节点特征工程、节点表示学习（图嵌入）、标签传播（消息传递）、图神经网络

- 半监督节点分类的思路：
    1. 给出$n$个节点的邻接矩阵$A$
    2. 给出矩阵的标签向量$Y=\{0, 1\}^n$，其中$Y_v = 1$属于类别1，$Y_v = 0$属于类别0
    3. 目标是预测未标签的节点分类

- 半监督节点分类问题的求解方法对比

| 方法 | 图嵌入 | 表示学习 | 使用属性特征 | 使用标注 | 直推式 | 归纳式 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 人工特征工程 | 是 | 否 | 否 | 否 | / | / |
| 基于随机游走的方法 | 是 | 是 | 否 | 否 | 是 | 否 |
| 基于矩阵分解的方法 | 是 | 是 | 否 | 否 | 是 | 否 |
| 标签传播 | 否 | 否 | 是/否 | 是 | 是 | 否 |
| 图神经网络 | 是 | 是 | 是 | 是 | 是 | 是 |

- 人工特征工程：节点重要度、集群系数、Graphlet等
- 基于随机游走的方法，构建自监督表示学习任务实现图嵌入，无法泛化到新节点，例如DeepWalk、Node2Vec、LINE、SDNE等
- 标签传播：假设“物以类聚、人以群分”，利用领域节点类别预测当前节点类别，无法泛化到新节点，例如Label Propagation、Interative Classification、Belief Propagation、Correct & Smooth等
- 图神经网络：利用深度学习和神经网络，构建领域节点信息聚合计算图，实现节点嵌入和类别预测，可泛化到新节点，例如GCN、GraphSAGE、GAT、GIN等

- Label Propagation：消息传递机制，利用领域节点类别信息

- 标签传播和集体分类：
    - Label Propagation（Relational Classification）
    - Iterative Classification
    - Correct & Smooth
    - Belief Propagation
    - Masked Label Prediction

## 2 对图的基本假设

- homophily：具有相似属性特征的节点更可能相连且具有相同类别
- Influence：“近朱者赤，近墨者黑”

- 相似节点更可能相近，可以使用KNN最近邻分类方法
- 对一个节点进行标签分类，需要节点自身的属性特征、领域节点类别和属性特征

## 3 标签传播（Label Propagation）

- 算法步骤：
    1. 初始化所有节点，对已知标签设置为$Y_v=\{0,1\}$，未知标签设置为$Y_v = 0.5$
    2. 开始迭代，计算该节点周围的所有节点P值的总和的平均值（加权平均）
    $$
    P(Y_v = C) = \frac{1}{\displaystyle \sum_{(v, u) \in E} A_{v, u}} \sum_{(v, u) \in E} P(Y_u = c)
    $$
    3. 当节点P值都收敛之后，可以设定阈值，进行类别划分，例如大于0.5设置为类别1，小于0.5设置为类别0

- 算法缺点：
    - 仅使用网络连接信息，没有使用节点属性信息
    - 不保证收敛

## 4 Iterative Classification

- 定义：
    - 节点$v$的属性信息为$f_v$，领域节点$N_v$的类别为$z_v$
    - 分类器$\phi_1(f_v)$仅使用节点属性特征$f_v$
    - 分类器$\phi_2(f_v, z_v)$使用节点属性特征$f_v$和网络连接特征$z_v$（即领域节点类别信息）
    - $z_v$为包含领域节点类别信息的向量

- 算法步骤：
    1. 使用已标注的数据训练两个分类器：$\phi_1(f_v)$、$\phi_2(f_v, z_v)$
    2. 迭代直至收敛：用$\phi_1$预测$Y_v$，用$Y_z$计算$z_v$，然后再用$\phi_2$预测所有节点类别，更新领域节点$z_v$，用新的$\phi_2$更新$Y_v$

- 算法缺点：不保证收敛

- 该算法可抽象为马尔科夫假设，即$P(Y_v) = P(Y_v | N_v)$

## 5 Correct & Smooth

- 算法步骤：
    1. 训练基础分类器，预测所有节点的分类结果，包含分类的类别概率（soft label）
    2. 得到训练好的基础分类器
    3. Correct步骤：计算training error，将不确定性进行传播和均摊（error在图中也有homophily）
    4. Smooth步骤：对最终的预测结果进行Label Propagation

- Correct步骤：
    - 扩散training errors $E^{(0)}$
    - 将邻接矩阵$A$进行归一化，得到$ \tilde{A} = D^{-\frac{1}{2}} A D^{\frac{1}{2}}$
    - 迭代计算：$E^{(t+1)} \leftarrow (1 - \alpha) \cdot E^{(t)} + \alpha \cdot \tilde{A} E^{(t)}$

> $\tilde{A}$特性：
> 1. $\tilde{A}$的特征值$\lambda \in [-1, 1]$
> 2. 幂运算之后，依然保证收敛
> 3. 如果$i$和$j$节点相连，则$\displaystyle \tilde{A}_{ij} = \frac{1}{\sqrt{d_i} \sqrt{d_j}}$

- Smooth步骤（类似Label Propagation）：
    - 通过图得到$Z^{(0)}$
    - 迭代计算：$Z^{(t+1)} \leftarrow (1 - \alpha) \cdot Z^{(t)} + \alpha \cdot \tilde{A} Z^{(t)}$

其中，$Z$中的每个节点的分类概率并不是求和为1的

- 总结：
    1. Correct & Smooth(C&S)方法用图的结构信息进行后处理
    2. Correct步骤对不确定性（error）进行扩散
    3. Smooth步骤对最终的预测结果进行扩散
    4. C&S是一个很好的半监督节点分类方法

## 6 Loopy Belief Propagation

- 类似消息传递，基于动态规划，即下一时刻的状态仅取决于上一时刻，当所有节点达到共识时，可得最终结果

- 算法思路：
    1. 定义一个节点序列
    2. 按照边的有向顺序排列
    3. 从节点$i$到节点$i+1$计数（类似报数）

- 定义：
    - Label-label potential matrix $\psi$：当邻居节点$i$为类别$Y_i$时，节点$j$为类别$Y_j$的概率（标量），反映了节点和其邻居节点之间的依赖关系
    - Prior belief $\phi$：$\phi(Y_i)$表示节点$i$为类别$Y_i$的概率
    - $m_{i \rightarrow j}(Y_j)$：表示节点$i$认为节点$j$是类别$Y_j$的概率
    - $\mathcal{L}$：表示节点的所有标签

- 算法步骤：
    1. 初始化所有节点信息都为1
    2. 迭代计算：
    $$
    m_{i \rightarrow j}(Y_j) = \sum_{Y_i \in \mathcal{L}} \psi(Y_i, Y_j) \phi_i(Y_i) \prod_{k \in N_j \backslash j} m_{k \rightarrow i} (Y_i), \ \forall Y_j \in \mathcal{L}
    $$
    3. 收敛之后，可得到结果：
    $$
    b_i(Y_i) = \phi_i(Y_i) \prod_{j \in N_i} m_{j \rightarrow i} (Y_i), \ \forall Y_j \in \mathcal{L}
    $$

- 优点：易于编程实现，可以应用于高阶$\psi(Y_i, Y_j, Y_k, Y_v \dots)$
- 存在问题：如果图中有环，会不收敛

## 7 Masked Label Prediction

- 灵感来自于语言模型BERT

- 算法思路：
    1. 随机将节点的标签设置为0，用$[X, \tilde{Y}]$预测已标记的节点标签
    2. 使用$\tilde{Y}$预测未标注的节点

## 8 本章总结

本次任务，主要介绍了标签传播与节点分类相关内容，包括：

- 介绍半监督节点分类，并比较求解方法（人工特征方法、基于随机游走的方法、基于矩阵分解的方法、标签传播、图神经网络）
- 介绍5种标签传播和集体分类，包含Label Propagation、Iterative Classification、Correct & Smooth、Belief Propagation、Masked Label Prediction
