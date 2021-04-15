# Task08 Bagging的原理和案例分析

## 1 知识梳理

### 1.1 Bagging的思路
- 集成模型最后的预测结果
- 采用一定策略影响基模型训练，保证基模型服从一定的假设条件
- 通过不同的采样增加模型的差异性

### 1.2 Bagging的原理分析
- 核心：自助采样，即有放回抽样
- 基本流程：得到含有K次自助采样的T个样本集合，使用基模型对每个集合进行训练，得到基学习器，再进行融合即可
- 预测结果：
  - 对于回归问题的预测：预测结果是所有模型的预测结果取平均值
  - 对于分类问题的预测：预测结果是所有模型中出现最多的预测结果
- 适用性：
  - 降低方差的技术
  - 适用于高维小样本的数据集，可采用列采样的Bagging

## 2 实战练习

&emsp;&emsp;使用`sklearn.ensemble.BaggingRegressor`和`sklearn.ensemble.BaggingClassifier`，默认的基模型是树模型

### 2.1 决策树的建立过程
- 树的每个非叶子节点表示特征的判断；分支表示对样本的划分
- 节点划分使用的指标主要是信息增益和`Gini`指数

#### 2.1.1 信息增益(IG)
- 概念：划分前后信息不确定性程度的减小
- 信息不确定程度：使用信息熵来度量$\displaystyle H(Y)=-\sum p_i \log p_i$，其中$i$表示样本的类别，$p$表示该类样本出现的概率
- 对样本划分之后的条件熵：$\displaystyle H(Y|X)=\sum_{x \in X}p(X=x)H(Y|X=x)$
- 信息增益：信息熵与条件熵之差，$IG = H(Y) - H(Y|X)$
- 信息增益`IG`越大，该特征划分的数据信息量变化越大，样本的“纯度”越高

#### 2.1.2 `Gini`指数
- `Gini`指数用于衡量数据的“不纯度”：$\displaystyle Gini=1-\sum p_i^2$
- 对样本划分之后的`Gini`指数：$\displaystyle Gini_x = \sum_{x \in X} p(X=x)\left( 1-\sum p_i^2 \right)$
- `Gini`指数越小，该特征划分的数据信息量变化越大，样本的“纯度”越高

### 2.2 Bagging典型应用——随机森林
- 随机森林由许多决策树bagging组成的
- 采用随机采样构建决策树的特征
- 预测结果是由多个决策树输出的结果组合而成

### 2.3 使用`BaggingClassifier`构建模型

#### 2.3.1 创建自定义数据集
使用`sklearn.datasets.make_classification`方法创建数据集


```python
from sklearn.datasets import make_classification

# 生成特征数为20，其中多信息特征数为10，冗余信息特征数为5，总样本数为1000个的数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
n_redundant=5, random_state=5)
```

#### 2.3.2 对模型评估
&emsp;&emsp;使用`sklearn.model_selection.RepeatedStratifiedKFold`对模型进行10层交叉验证（其中重复3次）评估


```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
```


```python
model = BaggingClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, 
                           n_jobs=-1, error_score='raise')

print('模型准确率: %.3f，模型标准差：%.3f' % (np.mean(n_scores), np.std(n_scores)))
```

    模型准确率: 0.902，模型方差：0.033
    
