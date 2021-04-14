# Task10 前向分步算法与梯度提升决策树

## 1 知识梳理

### 1.1 前向分步算法
- 加法模型：求解以下函数的最优解$(\beta,\gamma)$
$$
\min _{\beta_{m}, \gamma_{m}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
$$
- 前向分布算法：
  1. 初始化$f_0(x)=0$
  2. 对于$m=1,2,\cdots,M$，其中$M$表示基本分类器的个数  
    - 极小化损失函数：$\displaystyle 
   \left(\beta_{m}, \gamma_{m}\right)=\arg \min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right)$，得到参数$\beta_{m}$与$\gamma_{m}$  
     - 更新：$ f_{m}(x)=f_{m-1}(x)+\beta_{m} b\left(x ; \gamma_{m}\right)$
  3. 得到加法模型：$\displaystyle f(x)=f_{M}(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)$
- Adaboost算法是前向分步算法的特例

### 1.2 梯度提升决策树（GBDT）

#### 1.2.1 基于残差学习的提升树算法
- 最佳划分点判断：
  - 分类树：用**纯度**来判断，使用信息增益（ID3算法），信息增益比（C4.5算法），基尼系数（CART分类树）
  - 回归树：用**平方误差**判断
- 用**每个样本的残差**修正样本权重以及计算每个基本分类器的权重
- 算法步骤：  
  1. 初始化$f_0(x) = 0$                        
  2. 对$m = 1,2,...,M$：                  
    - 计算每个样本的残差:$r_{m i}=y_{i}-f_{m-1}\left(x_{i}\right), \quad i=1,2, \cdots, N$  
    - 拟合残差$r_{mi}$学习一棵回归树，得到$T\left(x ; \Theta_{m}\right)$                        
    - 更新$f_{m}(x)=f_{m-1}(x)+T\left(x ; \Theta_{m}\right)$
  3. 得到最终的回归问题的提升树：$\displaystyle f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)$

#### 1.2.2 梯度提升决策树算法（GBDT）
- 算法思路：  
  使用最速下降法的近似方法，利用损失函数的负梯度在当前模型的值$-\left[\frac{\partial L\left(y, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}$，作为回归问题提升树算法中的残差的近似值，拟合回归树。
- 算法步骤：
  1. 初始化$f_{0}(x)=\arg \min _{c} \sum_{i=1}^{N} L\left(y_{i}, c\right)$                     
  2. 对于$m=1,2,\cdots,M$：                   
    - 对$i = 1,2,\cdots,N$计算：$r_{m i}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}$                
    - 对$r_{mi}$拟合一个回归树，得到第m棵树的叶结点区域$R_{m j}, j=1,2, \cdots, J$                           
    - 对$j=1,2,\cdots,J$，计算：$c_{m j}=\arg \min _{c} \sum_{x_{i} \in R_{m j}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+c\right)$                      
    - 更新$f_{m}(x)=f_{m-1}(x)+\sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$                    
  3. 得到回归树：$\hat{f}(x)=f_{M}(x)=\sum_{m=1}^{M} \sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$

## 2 实战练习

### 2.1 基于残差学习的提升树算法


```python
import numpy as np

X = np.arange(1, 11)
y = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])
```


```python
class Tree:
    def __init__(self, split_point, mse, left_value, right_value, residual):
        # feature最佳切分点
        self.best_split_point = split_point
        # 平方误差
        self.mse = mse
        # 左子树值
        self.left_value = left_value
        # 右子树值
        self.right_value = right_value
        # 每棵决策树生成后的残差
        self.residual = residual
```


```python
class ResidualBT:
    def __init__(self, X, y, tol=0.05, n_estimators=6):
        # 训练数据：实例
        self.X = X
        # 训练数据：标签
        self.y = y
        # 最大迭代次数
        self.n_estimators = n_estimators
        # 回归树
        self.T = []

    def fit(self):
        """
        对训练数据进行学习
        :return:
        """

        # 得到切分点
        split_point = self.split_point()

        residual = self.y.copy()
        for i in range(self.n_estimators):
            tree, residual = self.build_desicion_tree(split_point, residual)
            self.T.append(tree)

    def predict(self, X):
        """
        对新数据进行预测
        """
        m = np.shape(X)[0]
        y_predict = np.zeros(m)

        for tree in self.T:
            for i in range(m):
                if X[i] < tree.best_split_point:
                    y_predict[i] += tree.left_value
                else:
                    y_predict[i] += tree.right_value
        return y_predict

    def sse(self):
        """平方损失误差"""
        y_predict = self.predict(X)
        return np.sum((y_predict - y) ** 2)

    def score(self, X, y):
        """对训练效果进行评价"""
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
        return 1 - error_rate

    def split_point(self):
        """
        获取切分点
        :return: 切分点
        """
        return (self.X[0:-1] + self.X[1:]) / 2

    def build_desicion_tree(self, split_point, label):
        m_s_list = []
        c1_list = []
        c2_list = []
        for p in split_point:
            # 切分点左边的label值
            label_left = label[0:int(p)]
            # 切分点右边的label值
            label_right = label[int(p):]
            c1 = np.mean(label_left)
            c2 = np.mean(label_right)
            m_s = np.sum((label_left - c1) ** 2) + np.sum((label_right - c2) ** 2)
            c1_list.append(c1)
            c2_list.append(c2)
            m_s_list.append(m_s)
        # 得到m_s最小值所在的位置
        best_index = np.argmin(m_s_list)
        # 得到最优切分点
        best_split_point = split_point[int(best_index)]
        # 得到最优均方误差
        best_mse = m_s_list[int(best_index)]
        # 得到左子树的label值
        best_y_lf = label[0:int(best_split_point)]
        lf_value = np.mean(best_y_lf)
        # 得到右子树的label值
        best_y_rt = label[int(best_split_point):]
        rt_value = np.mean(best_y_rt)
        # 得到决策树的残差
        residual = np.concatenate((best_y_lf - lf_value, best_y_rt - rt_value,))
        tree = Tree(best_split_point, best_mse, lf_value, rt_value, residual)
        return tree, residual
```


```python
clf = ResidualBT(X, y, n_estimators=6)
clf.fit()
y_predict = clf.predict(X)
score = clf.score(X, y)
print("\n原始输出:", y)
print("预测输出:", y_predict)
print("预测正确率：{:.2%}".format(score))
print("平方损失误差：{:.2}".format(clf.sse()))
```

    
    原始输出: [5.56 5.7  5.91 6.4  6.8  7.05 8.9  8.7  9.   9.05]
    预测输出: [5.63       5.63       5.81831019 6.55164352 6.81969907 6.81969907
     8.95016204 8.95016204 8.95016204 8.95016204]
    预测正确率：94.58%
    平方损失误差：0.17
    

### 2.2 梯度提升决策树算法（GBDT）


```python
import pandas as pd

data = [[5, 20, 1.1],
     [7, 30, 1.3],
     [21, 70, 1.7],
     [30, 60, 1.8]]
df = pd.DataFrame(columns=['age', 'weight', 'height'], data=data)
X = df[['age', 'weight']].values
y = df['height'].values
```


```python
from sklearn.ensemble import GradientBoostingRegressor

# 学习率：learning_rate=0.1，迭代次数：n_trees=5，树的深度：max_depth=3
gbr = GradientBoostingRegressor(n_estimators=5, learning_rate=0.1,
    max_depth=3, random_state=0, loss='ls')
gbr.fit(X, y)
gbr.predict([[25, 65]])[0]
```




    1.56713975



## 3 练一练
GradientBoostingRegressor与GradientBoostingClassifier函数各个参数的意思

### 3.1 GradientBoostingRegressor各个参数的意思

- loss：{`ls`,`lad`,`huber`,`quantile`}, default=`ls`：  
  - `ls`：指最小二乘回归；   
  - `lad`：(最小绝对偏差) 是仅基于输入变量的顺序信息，具有高度鲁棒的损失函数；  
  - `huber`：上述两者的结合；  
  - `quantile`：允许分位数回归（用于alpha指定分位数）  
- learning_rate：学习率用于缩小每棵树的贡献learning_rate，在learning_rate和n_estimators之间需要权衡
- n_estimators：执行迭代次数
- subsample：用于拟合各个基学习器的样本比例。如果小于1.0，将使得随机梯度增强。subsample与参数n_estimators有关联，选择subsample<1.0会导致方差减少和偏差增加
- criterion：{`friedman_mse`，`mse`，`mae`}，默认为`friedman_mse`：`mse`是均方误差，`mae`是平均绝对误差。默认值`friedman_mse`通常是最好的，因为在大多情况下可以提供更好的近似值
- min_samples_split：默认为2，拆分内部节点所需的最少样本数
- min_samples_leaf：默认为1，在叶节点处需要的最小样本数
- min_weight_fraction_leaf：默认为0.0，在所有叶节点处（所有输入样本）的权重总和中的最小加权数。如果未提供sample_weight，则样本的权重相等
- max_depth：默认为3，各个回归模型的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量
- min_impurity_decrease：如果节点拆分会导致不纯度大于或等于该值，则该节点将被拆分。
- min_impurity_split：提前停止树生长的阈值。如果节点的不纯度高于该值，则该节点将拆分
- max_features {`auto`, `sqrt`, `log2`}，int或float：寻找最佳切分点时要考虑的特征个数：
  - 如果是int，则表示节点切分的特征个数
  - 如果是float，max_features则为小数，根据公式int(max_features * n_features)确定节点切分的特征个数
  - 如果是`auto`，则max_features=n_features
  - 如果是`sqrt`，则max_features=sqrt(n_features)
  - 如果为`log2`，则为max_features=log2(n_features)
  - 如果没有，则max_features=n_features

### 3.2 GradientBoostingClassifier各个参数的意思

- loss：{`deviance`,`exponential`}, default=`deviance`：
  - `deviance`是指对具有概率输出的分类（等同于logistic回归）
  - 对于`exponential`梯度提升方法，可等同于AdaBoost算法
- learning_rate：学习率用于缩小每棵树的贡献learning_rate，在learning_rate和n_estimators之间需要权衡
- n_estimators：执行迭代次数
- subsample：用于拟合各个基学习器的样本比例。如果小于1.0，将使得随机梯度增强。subsample与参数n_estimators有关联，选择subsample<1.0会导致方差减少和偏差增加
- criterion：{`friedman_mse`，`mse`，`mae`}，默认为`friedman_mse`：`mse`是均方误差，`mae`是平均绝对误差。默认值`friedman_mse`通常是最好的，因为在大多情况下可以提供更好的近似值
- min_samples_split：默认为2，拆分内部节点所需的最少样本数
- min_samples_leaf：默认为1，在叶节点处需要的最小样本数
- min_weight_fraction_leaf：默认为0.0，在所有叶节点处（所有输入样本）的权重总和中的最小加权数。如果未提供sample_weight，则样本的权重相等
- max_depth：默认为3，各个回归模型的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量
- min_impurity_decrease：如果节点拆分会导致不纯度大于或等于该值，则该节点将被拆分。
- min_impurity_split：提前停止树生长的阈值。如果节点的不纯度高于该值，则该节点将拆分
- max_features {`auto`, `sqrt`, `log2`}，int或float：寻找最佳切分点时要考虑的特征个数：
  - 如果是int，则表示节点切分的特征个数
  - 如果是float，max_features则为小数，根据公式int(max_features * n_features)确定节点切分的特征个数
  - 如果是`auto`，则max_features=n_features
  - 如果是`sqrt`，则max_features=sqrt(n_features)
  - 如果为`log2`，则为max_features=log2(n_features)
  - 如果没有，则max_features=n_features
