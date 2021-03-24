# Task02 掌握基本的回归模型

## 1 知识梳理

### 1.1 使用sklearn构建完整的机器学习项目流程
机器学习项目的步骤有如下几步：
1. 明确项目任务：回归/分类
2. 收集数据集并选择合适的特征。
3. 选择度量模型性能的指标。
4. 选择具体的模型并进行训练以优化模型。
5. 评估模型的性能并调参。

### 1.2选择度量模型性能的指标

其目的是通过各种指标（即得分）衡量模型的性能。  
衡量回归模式经常使用的指标如下：
- MSE均方误差：$\displaystyle \text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$
- MAE平均绝对误差:$\displaystyle \text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|$
- $R^2$决定系数：$R^2(y, \hat{y}) = 1 - \displaystyle \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
- 解释方差得分:$\displaystyle explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$

### 1.3 选择具体的模型并训练

#### 1.3.1 线性回归模型
- 回归分析：研究的是因变量（目标）和自变量（特征）之间的关系。这种技术通常用于预测分析，时间序列模型以及发现变量之间的因果关系
- 线性回归模型：假设目标值与特征之间线性相关，即满足一个多元一次方程。通过构建损失函数，来求解损失函数最小时的参数$w$
  - 最小二乘估计：$L(w)=w^T X^TXw-2w^T X^T Y + YY^T \Rightarrow \hat{w}=(X^TX)^{-1}X^TY$
  - 几何解释：平面$X$的法向量$Y-Xw$与平面$X$相互垂直，$X^T(Y-Xw)=0 \Rightarrow \hat{w}=(X^TX)^{-1}X^TY$
  - 概率视角：假设噪声$\epsilon \backsim N(0,\sigma^2),y=f(w)+\epsilon=w^Tx+\epsilon$，使用极大似然估计，$L(w) = log\;P(Y|X;w) \Rightarrow argmax_w L(w) = argmin_w[l(w) = \sum\limits_{i = 1}^{N}(y_i-w^Tx_i)^2]$

#### 1.3.2 线性回归的推广
  - 多项式回归：使用`sklearn.preprocessing.PolynomialFeatures`，可得$[a,b] \Rightarrow [1,a,b,a^2,ab,b^2]$，如果`interaction_only=True`，则输出$[1,a,b,ab]$ 
  - 广义可加模型(GAM)：使用`pygam.LinearGAM`

#### 1.3.3 回归树（决策树）
回归树：依据分层和分割的方式将特征空间划分为一系列简单的区域。对某个给定的待预测的自变量，用他所属区域中训练集的平均数或者众数对其进行预测
  - 优点：解释性强、更接近人的决策方式、用图来表示、直接做定性的特征、很好处理缺失值和异常值
  - 缺点：对异常值不敏感、预测准确性一般无法达到其他回归模型的水平

#### 1.3.4 支持向量机回归(SVR)
- 对偶问题：$f(x)$与$g_i(x)$为凸函数，$h_j(x)$为线性函数，X是凸集，$x^*$满足KKT条件，那么$D^* = P^*$
- 概念描述：落在$f(x)$的$\epsilon$邻域空间中的样本点不需要计算损失，这些都是预测正确的，其余的落在$\epsilon$邻域空间以外的样本才需要计算损失

## 2 实战练习：使用sklearn构建完整的回归项目

本例使用sklearn内置数据集：糖尿病数据集


```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use("ggplot")      
import seaborn as sns
```

### 2.1 选择合适的特征


```python
from sklearn import datasets
# sklearn内置数据集：糖尿病数据集
# 返回一个类似于字典的类
diabetes = datasets.load_diabetes() 
X = diabetes.data
y = diabetes.target
features = diabetes.feature_names
diabetes_data = pd.DataFrame(X,columns=features)
diabetes_data['disease_progression'] = y
diabetes_data.head()
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>disease_progression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>206.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>135.0</td>
    </tr>
  </tbody>
</table>
</div>



各个特征的相关解释：
- age：年龄
- sex：性别
- bmi：体重指标
- bp：平均血压
- s1、s2、s3、s4、s5、s6：六次血清测量值
- disease_progression：一年疾病进展的测量值

### 2.2 线性回归模型 


```python
# 引入线性回归方法
from sklearn import linear_model
# 创建线性回归的类
lin_reg = linear_model.LinearRegression()    
# 输入特征X和因变量y进行训练
lin_reg.fit(X,y)
# 输出模型的系数
print("糖尿病数据集的线性回归模型系数：",lin_reg.coef_)
# 输出模型的决定系数R^2
print("糖尿病数据集的线性回归模型得分：",lin_reg.score(X,y))    
```

    糖尿病数据集的线性回归模型系数： [ -10.01219782 -239.81908937  519.83978679  324.39042769 -792.18416163
      476.74583782  101.04457032  177.06417623  751.27932109   67.62538639]
    糖尿病数据集的线性回归模型得分： 0.5177494254132934
    

### 2.3 决策树模型


```python
from sklearn.tree import DecisionTreeRegressor    
reg_tree = DecisionTreeRegressor(criterion = "mse",min_samples_leaf = 5)
reg_tree.fit(X,y)
print("糖尿病数据集的决策树模型得分：",reg_tree.score(X,y))
```

    糖尿病数据集的决策树模型得分： 0.7617420924706106
    

### 2.4 SVR模型


```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler     # 标准化数据
from sklearn.pipeline import make_pipeline   

# 使用管道，把预处理和模型形成一个流程
reg_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
reg_svr.fit(X, y)
print("糖尿病数据集的SVR模型得分：",reg_svr.score(X,y))
```

    糖尿病数据集的SVR模型得分： 0.20731089959670035
    
