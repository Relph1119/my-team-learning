# Task02 图的基本表示和特征工程

## 1 复习

- 图是描述大自然的通用语言，图数据自带关联结构
- 图神经网络是端到端的表示学习，可以自动学习特征，无需人为的特征工程
- 不同的任务类型：包括Graph层、Node层、Community(subgraph)层、Edge层
- 图机器学习可以和人工智能各个方向进行结合

## 2 图的基本表示

### 2.1 图的基本表示

- Objects：nodes（节点）、vertices（顶点）表示为$N$
- Interactions（关系）：links、edges表示为$E$
- System：network、graph表示为$G(N,E)$

### 2.2 设计本体图

- 如何设计本体图，取决于将来想解决什么样的问题
- 有些时候，本体图是唯一的、无歧义的

### 2.3 图的类型

- 无向图：连接是无方向的
- 有向图：连接是有方向的
- 异质图：节点和连接都存在不同的类型
- 二分图（Bipartite Graph）
    - 节点只有两类
    - 展开二分图：将连接了另一类的节点进行分别连接

### 2.4 节点连接数

- 无向图：平均度为 $\displaystyle \bar{k} = \langle k \rangle = \frac{1}{N} \sum_{i=1}^N k_i = \frac{2E}{N} $
- 有向图：平均度为 $\displaystyle \bar{k} = \frac{E}{N}$

### 2.5 邻接矩阵Adjacency Matrix

- 无向图：
    - 邻接矩阵是对称阵 $A_{ij} = A_{ji}$，主对角线为0，$A_{ii} = 0$
    - 连接总数：$\displaystyle L = \frac{1}{2} \sum_{i=1}^N k_i = \frac{1}{2} \sum_{ij}^N A_{ij}$
    - 沿行或列求和均可：$k_i = \sum_{j=1}^N A_{ij}$，$k_j = \sum_{i=1}^N A_{ij}$

- 有向图
    - 邻接矩阵是非对称矩阵，$A_{ij} \neq A_{ji}$，主对角线为0，$A_{ii} = 0$
    - 连接总数：$\displaystyle L = \sum_{i=1}^N k_i^{in} = \sum_{j=1}^N k_j^{out} = \sum_{ij}^N A_{ij}$
    - $k_i^{out} = \sum_{j=1}^N A_{ij}$
    - $k_j^{in} = \sum_{i=1}^N A_{ij}$

### 2.6 连接列表、邻接列表

- 连接列表：只记录存在连接的节点对
- 邻接列表：只记录相关连接的节点，每个节点占用一行

### 2.7 其他类型的图

- Unweighted（无权图、无向图）
    - $A_{ii} = 0$
    - $A_{ij} = A_{ji}$
    - $\displaystyle E = \frac{1}{2} \sum_{i,j=1}^N A_{ij}$
    - $\displaystyle \bar{k} = \frac{2E}{N}$

- Weighted（有权图、无向图）
    - $A_{ii} = 0$
    - $A_{ij} = A_{ji}$
    - $\displaystyle E = \frac{1}{2} \sum_{i,j=1}^N \text{nonzero}(A_{ij})$
    - $\displaystyle \bar{k} = \frac{2E}{N}$

- Self-edges(self-loops)（带自循环的图，无向图）
    - $A_{ii} \neq 0$
    - $A_{ij} = A_{ji}$
    - $\displaystyle E = \frac{1}{2} \sum_{i,j=1, i \neq j}^N A_{ij} + \sum_{i=1}^N A_{ii}$

- Multigraph（多种通路，无向图）
    - $ A_{ii} = 0 $
    - $ A_{ij} = A_{ji} $
    - $ \displaystyle E = \frac{1}{2} \sum_{i,j = 1}^N \text{nonzero}(A_{ij}) $
    - $ \displaystyle \bar{k} = \frac{2E}{N} $

### 2.8 图的连通性

- Connected graph（无向图）：如果能满足任意两个节点能到达的图
- Disconnected graph（有向图）：不能满足任意两个节点能到达的图
- Giant Component：最大连通域
- Isolated node：孤立节点

- Strongly connected directed graph：有向图中，任意两个节点可以互相到达
- Weakly connected directed graph：忽略方向之后，如果任意两个节点能互相到达
- Strongly connected compoinents(SCCs)：强连通性
- In-component：节点指向SCC
- Out-component：节点指出SCC

## 3 传统图机器学习（人工特征工程+机器学习）

### 3.1 特征分类

- 属性特征：Weight、Ranking、Type、Sign、多模态特征（图像、视频、文本、音频）
- 连接特征

### 3.2 传统机器学习步骤

1. 把节点、连接、全图变成D维向量
2. 将D维向量输入到机器学习模型中进行训练
3. 给出一个新的图（节点、连接、全图）和特征，进行预测

### 3.3 节点层面的特征工程

- 主要流程：给出$G=(V,E)$，学习$f: V \rightarrow \mathbb{R}$
- 半监督节点分类任务：使用已知节点图，预测未知节点的类别
- 节点特征：Node degree、Node centrality（节点重要度）、Clustering coefficient（聚集系数）、Graphlets（子图模式）

**节点重要度（Node centrality）：**

1. Eigenvector centrality

公式：$\displaystyle c_v = \frac{1}{\lambda} \sum_{u \in N(v)} c_u$，$v$节点邻居节点的重要度求和

等价求解：$\lambda c = A c$，求邻接矩阵的特征值和特征向量

2. Betweenness centrality：一个节点是否处于交通要道上

公式：$\displaystyle c_v = \sum_{s \neq v \neq t} \frac{\#(\text{shortest paths between } s \text{ and } t \text{ than contain } v)}{\#(\text{shortest paths between } s \text{ and } t)}$

其中，分子表示其中有多少对节点的最短距离通过节点$v$；分母表示除了$v$节点之外，两两节点对数（对于连通域，分母都相同）

3. Closeness centrailty：去哪儿都近

公式：$\displaystyle c_v = \frac{1}{\displaystyle \sum_{u \neq v} \text{shortest path length between } u \text{ and } v}$

其中，分母表示节点$v$到其他节点$u$最短路径长度求和

**聚集系数（Clustering coefficient）：**

- 衡量节点有多抱团
- 公式：$\displaystyle e_v = \frac{\#(\text{edges among neighboring nodes})}{
\left ( \begin{array}{c}
k_v \\
2 
\end{array} \right )} \in [0, 1]$

其中，分子表示$v$节点相邻节点两两也相连个数（三角形个数），分母表示$v$节点相邻节点两两对数

- ego-network（自我中心网络）：计算三角形个数

**子图（Graphlets）：**

- Graphlets Degree Vector（GDV）计算方式：提取某一个节点的其余节点的子图个数，可以构建一个向量
- 5个节点的graphlets有73个子图个数，GDV具有73维的向量

### 3.4 连接层面的特征工程

- 通过已知连接补全未知连接
- 方案：直接提取link的特征，把link变成D维向量
- Link Prediction：
    - 随时间不变的（蛋白质分子）：随机删除一些连接，然后再预测
    - 随时间变化的（论文引用、社交网络）：使用上一个时间区段的图预测下一个时间区段的L个连接，选出Top N个连接，将这些连接和下一个真实时刻的连接进行比较，用于分析模型性能

- 具体步骤：
    1. 提取连接特征，转变为D维向量，计算分数 $c(x,y)$
    2. 将分数进行从高到低降序排列
    3. 获取Top N个连接
    4. 与真实连接进行比较，并分析算法性能

**连接特征分类：**

1. 基于两节点距离（Distance-based feature）

只看最短路径长度，忽略个数

2. 基于两节点局部连接信息（Local neighborhood overlap）

共同好友个数（Common neighbors）：$|N(v_1) \cap N(v_2) |$

交并比（Jaccard's coefficient）：$\displaystyle \frac{|N(v_1) \cap N(v_2)|}{|N(v_1) \cup N(v_2)|}$

共同好友是不是社牛（Adamic-Adar index）：$\displaystyle \sum_{u \in N(v_1) \cap N(v_2)} \frac{1}{\log (k_u)}$

3. 基于两节点在全图的连接信息（Global neighborhood overlap）

Katz index：节点$u$和节点$v$之间长度为$K$的路径个数

计算方式：$P^{(K)} = A^k$

例如：计算$P_{uv}^{(2)} = \sum_i A_{ui} * P_{iv}^{(1)} = \sum_i A_{ui} * A_{iv} = A_{uv} ^2$

 节点 $u$ 和节点 $v$ 之间长度为 $K$ 的路径个数是 $A_{uv}^K$ 矩阵的第 $u$ 行第 $v$ 列元素。

$\displaystyle S_{v_{1}v_{2}} = \sum_{l=1}^{\infty} \beta^l A_{v_{1}v_{2}}^l$，其中$\beta$为折减系数，$0 < \beta < 1$

最后，$\displaystyle S=\sum_{i=1}^{\infty} \beta^i A^i = (I - \beta A)^{-1} - I$

### 3.5 全图层面的特征工程

- 目标：提取出的特征应反映全图结构特点
- 使用Bag-of-Words（BoW）方法，可以应用于图，即将图看成文章，将节点看作单词，但是无法表示连接
- 区别：计算全图的Graphlet个数，而非特定节点领域Graphlet个数
- 公式：$(f_G)_i = \#(g_i \subseteq G) \text{ for } i = 1,2,\cdots, n_k$，表示第$i$个Graphlet在全图中的个数

**Graphlet Kernel：**

- 两个图的graphlet kernel：$K(G, G') = f_G^T f_{G'} $，两个图的Graphlet Count Vector数量积
- 计算上述公式太过复杂，该问题是NP-hard

**Weisfeiler-Lehman Kernel 算法：**

1. 对每个阶段上色，表示如下
$$
c^{(k+1)}(v) = \text{HASH} \left( \left \{ c^{(k)}(v), \{ c^{(k)}(u)\}_{u \in N(v)} \right \} \right)
$$

2. 重复上述步骤，进行$K$步，$c^{(K)}(v)$表示 K-hop 的连接

WL kernel 在计算上非常高效，复杂度是线性的，只需要关注非零元素即可

**Kernel Methods：**

- 可以表示两个图的数量积运算，即$K(G, G') = \phi(G)^T \phi(G')$

## 4 本章总结

本次任务，主要讲解了图的基本表示和特征工程，包括：

- 图的基本表示，包括无向图、有向图、二分图、有权图、邻接矩阵、图的连通性
- 节点层面的特征工程：Node degree（节点连接度）、Node centrality（节点重要度）、Clustering coefficient（聚集系数）、Graphlets（子图模式）
- 连接层面的特征工程：基于两节点距离、基于两节点局部连接信息、基于两节点在全图的连接信息
- 全图层面的特征工程：Graphlet Kernel、Weisfeiler-Lehman Kernel、Kernel Methods
