# Task03 掌握偏差与方差理论

## 1 知识梳理：优化基础模型

### 1.1 训练均方误差与测试均方误差
- 最常用的评价指标为均方误差：$\displaystyle \text{MSE} = \frac{1}{N}\sum\limits_{i=1}^{N}(y_i -\hat{ f}(x_i))^2$
- 如果所用的数据是训练集上的数据，那么这个误差为训练均方误差；如果使用测试集的数据计算均方误差，那么则称为测试均方误差
- 一般在训练误差达到最小时，测试均方误差一般很大！容易出现过拟合

### 1.2 偏差-方差的权衡
$$E\left(y_0-\hat{f}(x_0)\right)^2 = \text{Var}\left(\hat{f}(x_0)\right)+\left[\text{Bias}\left(\hat{f}(x_0)\right)\right]^2+\text{Var}(\varepsilon)$$

- $\text{Var}(\varepsilon)$称为建模任务的难度，也叫做不可约误差
- 模型的方差：用不同的数据集去估计$f$时，估计函数的改变量
- 一般来说，模型的复杂度越高，$f$的方差就会越大
- 模型的偏差：为了选择一个简单的模型去估计真实函数时所带入的误差
- 偏差度量的是单个模型的学习能力，而方差度量的是同一个模型在不同数据集上的稳定性（即鲁棒性）
- 目标：偏差和方差都需要小，才能使得测试均方差最小

### 1.3 特征提取
- 训练误差修正：构造一个特征较多的模型，加入关于特征个数的惩罚，从而对训练误差进行修正  
  $C_p = \frac{1}{N}(RSS  +  2d\hat{\sigma}^2)$，其中$d$为模型特征个数，$RSS = \sum\limits_{i=1}^{N}(y_i-\hat{f}(x_i))^2$，$\hat{\sigma}^2$为模型预测误差的方差的估计值，即残差的方差。
- 交叉验证：$K$折交叉验证是重复$K$次取平均值得到测试误差的一个估计$\displaystyle CV_{(K)} = \frac{1}{K}\sum\limits_{i=1}^{K}\text{MSE}_i$
- 最优子集选择、向前逐步选择：通过计算RSS进行迭代，每次选择RSS值最小的模型，最后选择测试误差最小的模型作为最优模型

### 1.4 压缩估计（正则化）
- 岭回归(L2正则化的例子)：在线性回归的损失函数的基础上添加对系数的约束或者惩罚$\lambda\sum\limits_{j=1}^p w_j^2$，通过牺牲线性回归的无偏性降低方差，有可能使得模型整体的测试误差较小，提高模型的泛化能力（**无偏性**的直观意义是样本估计量的数值在参数的真值附近摆动）
- Lasso回归(L1正则化的例子)：使用系数向量的L1范数替换岭回归中的L2范数
- 由于Lasso回归的RSS曲线与坐标轴相交时，回归系数中的某一个系数会为0，这样就能实现特征提取

### 1.5 降维
- 主成分分析（PCA）：通过最大投影方差将原始空间进行重构，即由特征相关重构为特征无关，即落在某个方向上的点(投影)的方差最大

## 2 实战练习

上接Task02，本例使用sklearn内置数据集：糖尿病数据集


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



### 2.1 向前逐步回归


```python
#定义向前逐步回归函数
def forward_select(data,target):
    variate=set(data.columns)  #将字段名转换成字典类型
    variate.remove(target)  #去掉因变量的字段名
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')  #目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    #循环筛选变量
    while variate:
        aic_with_variate=[]
        for candidate in variate:  #逐个遍历自变量
            formula="{}~{}".format(target,"+".join(selected+[candidate]))  #将自变量名连接起来
            aic=ols(formula=formula,data=data).fit().aic  #利用ols训练模型得出aic值
            aic_with_variate.append((aic,candidate))  #将第每一次的aic值放进空列表
        aic_with_variate.sort(reverse=True)  #降序排序aic值
        best_new_score,best_candidate=aic_with_variate.pop()  #最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
        if current_score>best_new_score:  #如果目前的aic值大于最好的aic值
            variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
            selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
            current_score=best_new_score  #最新的分数等于最好的分数
            print("aic is {},continuing!".format(current_score))  #输出最小的aic值
        else:
            print("for selection over!")
            break
    formula="{}~{}".format(target,"+".join(selected))  #最终的模型式子
    print("final formula is {}".format(formula))
    model=ols(formula=formula,data=data).fit()
    return(model)
```


```python
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型
forward_select(data=diabetes_data,target='disease_progression')
```

    aic is 4912.038220667561,continuing!
    aic is 4828.398482363347,continuing!
    aic is 4813.225718253229,continuing!
    aic is 4804.962491886372,continuing!
    aic is 4800.083415059462,continuing!
    aic is 4788.602540139351,continuing!
    for selection over!
    final formula is disease_progression~bmi+s5+bp+s1+sex+s2
    




    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x19ea2016860>




```python
lm=ols("disease_progression~bmi+s5+bp+s1+sex+s2",data=diabetes_data).fit()
lm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>disease_progression</td> <th>  R-squared:         </th> <td>   0.515</td>
</tr>
<tr>
  <th>Model:</th>                    <td>OLS</td>         <th>  Adj. R-squared:    </th> <td>   0.508</td>
</tr>
<tr>
  <th>Method:</th>              <td>Least Squares</td>    <th>  F-statistic:       </th> <td>   76.95</td>
</tr>
<tr>
  <th>Date:</th>              <td>Thu, 18 Mar 2021</td>   <th>  Prob (F-statistic):</th> <td>3.01e-65</td>
</tr>
<tr>
  <th>Time:</th>                  <td>08&#58;14&#58;44</td>       <th>  Log-Likelihood:    </th> <td> -2387.3</td>
</tr>
<tr>
  <th>No. Observations:</th>       <td>   442</td>        <th>  AIC:               </th> <td>   4789.</td>
</tr>
<tr>
  <th>Df Residuals:</th>           <td>   435</td>        <th>  BIC:               </th> <td>   4817.</td>
</tr>
<tr>
  <th>Df Model:</th>               <td>     6</td>        <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>       <td>nonrobust</td>      <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  152.1335</td> <td>    2.572</td> <td>   59.159</td> <td> 0.000</td> <td>  147.079</td> <td>  157.188</td>
</tr>
<tr>
  <th>bmi</th>       <td>  529.8730</td> <td>   65.620</td> <td>    8.075</td> <td> 0.000</td> <td>  400.901</td> <td>  658.845</td>
</tr>
<tr>
  <th>s5</th>        <td>  804.1923</td> <td>   80.173</td> <td>   10.031</td> <td> 0.000</td> <td>  646.617</td> <td>  961.767</td>
</tr>
<tr>
  <th>bp</th>        <td>  327.2198</td> <td>   62.693</td> <td>    5.219</td> <td> 0.000</td> <td>  204.001</td> <td>  450.439</td>
</tr>
<tr>
  <th>s1</th>        <td> -757.9379</td> <td>  160.435</td> <td>   -4.724</td> <td> 0.000</td> <td>-1073.262</td> <td> -442.614</td>
</tr>
<tr>
  <th>sex</th>       <td> -226.5106</td> <td>   59.857</td> <td>   -3.784</td> <td> 0.000</td> <td> -344.155</td> <td> -108.866</td>
</tr>
<tr>
  <th>s2</th>        <td>  538.5859</td> <td>  146.738</td> <td>    3.670</td> <td> 0.000</td> <td>  250.182</td> <td>  826.989</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.187</td> <th>  Durbin-Watson:     </th> <td>   2.043</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.552</td> <th>  Jarque-Bera (JB):  </th> <td>   1.172</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.016</td> <th>  Prob(JB):          </th> <td>   0.557</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.750</td> <th>  Cond. No.          </th> <td>    85.7</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### 2.2 岭回归

使用`sklearn.linear_model.ridge_regression`


```python
from sklearn import linear_model
reg_rid = linear_model.Ridge(alpha=.5)
reg_rid.fit(X,y)
print("糖尿病数据集的岭回归模型得分：",reg_rid.score(X,y))    
```

    糖尿病数据集的岭回归模型得分： 0.48750163913323585
    

### 2.3 Lasso回归
使用`sklearn.linear_model.Lasso`


```python
from sklearn import linear_model
reg_lasso = linear_model.Lasso(alpha = 0.5)
reg_lasso.fit(X,y)
print("糖尿病数据集的岭回归模型得分：",reg_lasso.score(X,y)) 
```

    糖尿病数据集的岭回归模型得分： 0.45524148827340677
    
