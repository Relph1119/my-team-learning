# Task04 掌握回归模型的评估及超参数调优

## 1 知识梳理

### 1.1 参数与超参数
- 参数：**使用**最小二乘法或者梯度下降法等最优化算法优化出来的数
- 超参数：**无法使用**最小二乘法或者梯度下降法等最优化算法优化出来的数
- 对比：
|序号|参数|超参数|
|:---:|:---|:---|
|1|定义了可使用的模型|用于帮助估计模型参数|
|2|从数据估计中得到|模型外部的配置，其值无法从数据中估计|
|3|不由编程者手动设置|由人工指定，可以使用启发式设置|
|4|保存为学习模型的一部分|被调整为给定的预测建模问题|

### 1.2 网格搜索
- 使用`sklearn.model_selection.GridSearchCV`
- 搜索思路：把所有的超参数选择都列出来，分别进行排列组合，针对每组超参数建立一个模型，选择测试误差最小的那组超参数。

### 1.3 随机搜索
- 使用`sklearn.model_selection.RandomizedSearchCV`
- 随机搜索法结果比稀疏化网格法稍好(有时候也会极差，需要权衡)
- 参数的随机搜索中的每个参数都是从可能的参数值的分布中采样的
- 优点：可以独立于参数数量和可能的值来选择计算成本，添加不影响性能的参数不会降低效率

## 2 实战练习

使用sklearn内置数据集：糖尿病数据集

### 2.1 对未调参的SVR进行评价


```python
from sklearn.svm import SVR     # 引入SVR类
from sklearn.pipeline import make_pipeline   # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler # 由于SVR基于距离计算，引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV  # 引入网格搜索调优
from sklearn.model_selection import cross_val_score # 引入K折交叉验证
from sklearn import datasets
import numpy as np 
```


```python
# sklearn内置数据集：糖尿病数据集
diabetes = datasets.load_diabetes() 
X = diabetes.data
y = diabetes.target
features = diabetes.feature_names
pipe_SVR = make_pipeline(StandardScaler(), SVR())
# 10折交叉验证
score1 = cross_val_score(estimator=pipe_SVR, X = X, y = y, scoring = 'r2', cv = 10)       
print("CV accuracy: %.3f +/- %.3f" % ((np.mean(score1)),np.std(score1)))
```

    CV accuracy: 0.151 +/- 0.040
    

### 2.2 使用网格搜索来对SVR调参


```python
from sklearn.pipeline import Pipeline
pipe_svr = Pipeline([("StandardScaler",StandardScaler()), ("svr",SVR())])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
# 注意__是指两个下划线，一个下划线会报错的
param_grid = [{"svr__C":param_range,"svr__kernel":["linear"]},  
              {"svr__C":param_range,"svr__gamma":param_range,"svr__kernel":["rbf"]}]
# 10折交叉验证
gs = GridSearchCV(estimator=pipe_svr, param_grid = param_grid, scoring = 'r2', cv = 10)
gs = gs.fit(X,y)
print("网格搜索最优得分：",gs.best_score_)
print("网格搜索最优参数组合：\n",gs.best_params_)
```

    网格搜索最优得分： 0.4741539611612783
    网格搜索最优参数组合：
     {'svr__C': 100.0, 'svr__gamma': 0.01, 'svr__kernel': 'rbf'}
    

### 2.3 使用随机搜索来对SVR调参


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform  # 引入均匀分布设置参数
pipe_svr = Pipeline([("StandardScaler",StandardScaler()), ("svr",SVR())])
# 构建连续参数的分布
distributions = dict(svr__C=uniform(loc=1.0, scale=4),    
                     svr__kernel=["linear","rbf"],         # 离散参数的集合                          
                     svr__gamma=uniform(loc=0, scale=4))
# 10折交叉验证
rs = RandomizedSearchCV(estimator=pipe_svr, param_distributions = distributions, scoring = 'r2', cv = 10)       
rs = rs.fit(X,y)
print("随机搜索最优得分：",rs.best_score_)
print("随机搜索最优参数组合：\n",rs.best_params_)
```

    随机搜索最优得分： 0.4638994569085143
    随机搜索最优参数组合：
     {'svr__C': 1.3067065834059988, 'svr__gamma': 2.684128235792981, 'svr__kernel': 'linear'}
    
