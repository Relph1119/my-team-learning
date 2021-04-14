# Task05 掌握基本的分类模型

## 1 知识梳理

### 1.1 选择度量模型性能的指标
- 与回归指标的差异：
  - 分类问题的因变量是离散变量，只衡量预测值和因变量的相似度是不可行的
  - 对于每个类别错误率的代价不同
- 分类的各种情况：
  - 真阳性TP：预测值和真实值都为正例；                        
  - 真阴性TN：预测值与真实值都为负例；                     
  - 假阳性FP：预测值为正，实际值为负；
  - 假阴性FN：预测值为负，实际值为正；
- 分类模型的指标
  - 准确率：$\displaystyle ACC = \frac{TP+TN}{FP+FN+TP+TN}$
  - 精度：$\displaystyle PRE = \frac{TP}{TP+FP}$
  - 召回率：$\displaystyle REC = \frac{TP}{TP+FN}$
  - F1值：$\displaystyle F1 = 2 \frac{PRE\times REC}{PRE + REC}$
  - ROC曲线：以假阳率为横轴，真阳率为纵轴画出来的曲线，曲线下方面积越大越好

### 1.2 选择具体的模型并训练

#### 1.2.1 逻辑回归
- 如果线性回归模型为$Y=\beta_0 + \beta_1 X$，`logistic`函数为${p(X) = \dfrac{e^{\beta_0 + \beta_1X}}{1+e^{\beta_0 + \beta_1X}}}$
- 通过假设数据服从0-1分布，计算$P(Y|X)$的极大似然估计：$\arg\max\limits_w \log P(Y|X)$，使用梯度下降法，计算$w_k^{(t+1)}\leftarrow w_k^{(t)} - \eta \sum\limits_{i=1}^{N}(y_i-\sigma(z_i))x_i^{(k)}$，其中$x_i^{(k)}$为第$i$个样本第$k$个特征

### 1.3 基于概率的分类模型

#### 1.3.1 线性判别分析
- 贝叶斯定理：${P(Y=k|X=x) = \dfrac{{\pi}_kf_k(x)}{\sum\limits_{l=1}^K{\pi}_lf_l(x)}}$ 
- 通过贝叶斯定理计算分子部分${\pi}_kf_k(x)$，比较分子值最大的哪个类别即为最终的类别，模型如下：  
  $${\begin{cases}\delta_k(x) = ln(g_k(x))=ln\pi_k+\dfrac{\mu}{\sigma^2}x-\dfrac{\mu^2}{2\sigma^2}\\{\hat{\mu}_k =\dfrac{1}{n_k}\sum\limits_{i:y_i=k}x_i}\\{\hat{\sigma}^2 =\dfrac{1}{n-K}\sum\limits_{k=1}^K\sum\limits_{i:y_i=k}(x_i-\hat{\mu}_k)^2}\end{cases}}$$通过计算${\delta_k(x)}$ ，可得到${k}$对应的${\delta_k(x)}$值大的类为最终的类别
- 降维分类：降维后的数据，同一类别的数据自身内部方差小，不同类别之间的方差大，简称为“类内方差小，类间方差大”

#### 1.3.2 朴素贝叶斯
- 朴素贝叶斯算法，将线性判别分析中的协方差矩阵的协方差全部变为0，只保留各自特征的方差，即假设各个特征之间是不相关的

### 1.4 决策树
- 给定一个观测值，因变量的预测值为它所属的终端结点内训练集的最常出现的类
- 使用分类错误率（$E = 1-max_k(\hat{p}_{mk})$）作为确定分裂结点的准则，但是在构建决策树时不够准确
- 基尼系数：$G = \sum\limits_{k=1}^{K} \hat{p}_{mk}(1-\hat{p}_{mk})$，如果取值小，意味着某个节点包含的观测点几乎来自同一类，其分类树也叫做`CART`
- 交叉熵：$D = -\sum\limits_{k=1}^{K} \hat{p}_{mk}log\;\hat{p}_{mk}$，如果某个结点纯度越高，交叉熵越小

### 1.5 支持向量机SVM
- 找到最大间隔超平面，将数据分开，即找到一个分隔平面距离最近的观测点最远
- 根据距离超平面最近的点，可得到SVM模型的具体形式$$\begin{aligned}
\min _{w, b} & \frac{1}{2}\|w\|^{2} \\
\text { s.t. } & y^{(i)}\left(w^{T} x^{(i)}+b\right) \geq 1, \quad i=1, \ldots, n
\end{aligned}$$
将优化问题转换为拉格朗日问题：$$\mathcal{L}(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{n} \alpha_{i}\left[y^{(i)}\left(w^{T} x^{(i)}+b\right)-1\right]$$ 
对上述问题求最小值，得到$w$和$b$的解,代入上述方程中，可得：$$\mathcal{L}(w, b, \alpha)=\sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)}$$
可构造如下对偶问题：
$$\begin{aligned}
    \max _{\alpha} & W(\alpha)=\sum_{i=1}^{n} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{n} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle \\
    \text { s.t. } & \alpha_{i} \geq 0, \quad i=1, \ldots, n \\
    & \sum_{i=1}^{n} \alpha_{i} y^{(i)}=0
    \end{aligned}$$
可得到$b$的值：$$b^{*}=-\frac{\max _{i: y^{(i)}=-1} w^{* T} x^{(i)}+\min _{i: y^{(i)}=1} w^{* T} x^{(i)}}{2}$$

### 1.6 非线性支持向量机
- 将数据投影至更高的维度
- 引入核函数的目的：避免在高纬度空间计算内积的恐怖计算量，享受在高维空间线性可分
- 多项式核函数：$$
   K\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\left(\left\langle\mathbf{x}_{i}, \mathbf{x}_{j}\right\rangle+c\right)^{d}$$
- 高斯核函数：$$
   K\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\exp \left(-\frac{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}^{2}}{2 \sigma^{2}}\right)$$使用时，需要将特征标准化
- `Sigmoid`核函数：$$
   K\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\tanh \left(\alpha \mathbf{x}_{i}^{\top} \mathbf{x}_{j}+c\right)$$
- 余弦相似度核：$$
   K\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\frac{\mathbf{x}_{i}^{\top} \mathbf{x}_{j}}{\left\|\mathbf{x}_{i}\right\|\left\|\mathbf{x}_{j}\right\|}$$

## 2 实战练习：使用sklearn构建完整的分类项目 

本例使用sklearn内置数据集：葡萄酒识别数据集


```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use("ggplot")      
import seaborn as sns
```


```python
from sklearn import datasets
# sklearn内置数据集：葡萄酒识别数据集
wine = datasets.load_wine()
X = wine.data
y = wine.target
features = wine.feature_names
wine_data = pd.DataFrame(X, columns=features)
wine_data['target'] = y
wine_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



各个特征的相关解释：
   - alcohol：酒精
   - malic_acid：苹果酸
   - ash：灰
   - alcalinity_of_ash：灰的碱度
   - magnesium：镁
   - total_phenols：总酚
   - flavanoids：类黄酮
   - nonflavanoid_phenols：非类黄酮酚
   - proanthocyanins：原花青素
   - color_intensity：色彩强度
   - hue：色调
   - od280/od315_of_diluted_wines：稀释酒的OD280 / OD315
   - proline：脯氨酸

### 2.1 逻辑回归

使用`sklearn.linear_model.LogisticRegression`


```python
from sklearn.linear_model import LogisticRegression
log_wine = LogisticRegression(solver='liblinear')
log_wine.fit(X,y)
log_wine.score(X,y)
```




    0.9719101123595506



### 2.2 线性判别分析
使用`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_wine = LinearDiscriminantAnalysis()
lda_wine.fit(X,y)
lda_wine.score(X,y)
```




    1.0



### 2.3 朴素贝叶斯  
使用`sklearn.naive_bayes.GaussianNB`


```python
from sklearn.naive_bayes import GaussianNB
NB_wine = GaussianNB()
NB_wine.fit(X, y)
NB_wine.score(X,y)
```




    0.9887640449438202



### 2.4 决策树
使用`sklearn.tree.DecisionTreeClassifier`


```python
from sklearn.tree import DecisionTreeClassifier
tree_wine = DecisionTreeClassifier(min_samples_leaf=5)
tree_wine.fit(X,y)
tree_wine.score(X,y)
```




    0.949438202247191



### 2.5 支持向量机SVM
使用`sklearn.svm.SVC`


```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svc_wine = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_wine.fit(X, y)
svc_wine.score(X,y)
```




    1.0


