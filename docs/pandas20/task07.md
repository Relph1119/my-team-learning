# Task7 缺失数据 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 缺失值的统计与删除
- 缺失信息的统计：主要使用`isna().sum()`查看缺失的比例
- 删除缺失信息：使用`dropna()`函数，其中`thresh`参数表示非缺失值 没有达到这个数量的相应维度会被删除

### 1.2 缺失值的填充和插值
- `fillna`函数：`limit`参数表示连续缺失值的最大填充次数
- `interpolate`函数：`limit_direction`参数表示控制方向，`limit`参数表示连续缺失值的最大填充次数

### 1.3 Nullable类型
- `None`除了等于自己本身之外，与其他任何元素不相等
- `np.nan`与其他任何元素不相等
- 在使用`equals`函数比较两张表时，会自动跳过两张表都是缺失值的位置
- 在时间序列对象中，使用`pd.NaT`表示缺失值

### 1.4 缺失数据的计算和分组
- 进行`sum`和`prod`时，缺失数据不影响计算
- 当使用累计函数（例如`cumsum`）时，会自动跳过缺失值所处的位置
- 在使用`groupby`、`get_dumies`函数时，通过设置`dropna=False`，缺失值可以作为一个类别

## 2 练一练

### 2.1 第1题

对一个序列以如下规则填充缺失值：如果单独出现的缺失值，就用前后均值填充，如果连续出现的缺失值就不填充，即序列`[1, NaN, 3, NaN, NaN]`填充后为`[1, 2, 3, NaN, NaN]`，请利用`fillna`函数实现。（提示：利用`limit`参数）

**我的解答：**


```python
import pandas as pd
import numpy as np
```


```python
s = pd.Series([np.nan, 1, np.nan, 3, 4, np.nan, 4, np.nan, np.nan, 5])
s
```




    0    NaN
    1    1.0
    2    NaN
    3    3.0
    4    4.0
    5    NaN
    6    4.0
    7    NaN
    8    NaN
    9    5.0
    dtype: float64




```python
# 分别使用ffill和bfill，并limit限制为1，得到两个Series
s_ffill = s.fillna(method='ffill', limit=1)
s_bfill = s.fillna(method='bfill', limit=1)
```


```python
# 构造一个DataFrame
pd.DataFrame([s_ffill, s_bfill])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对DataFrame求mean，其中skipna表示求mean的时候不忽略nan
pd.DataFrame([s_ffill, s_bfill]).mean(axis=0, skipna=False)
```




    0    NaN
    1    1.0
    2    2.0
    3    3.0
    4    4.0
    5    4.0
    6    4.0
    7    NaN
    8    NaN
    9    5.0
    dtype: float64



## 3 练习

### 3.1 Ex1：缺失值与类别的相关性检验
在数据处理中，含有过多缺失值的列往往会被删除，除非缺失情况与标签强相关。下面有一份关于二分类问题的数据集，其中`X_1, X_2`为特征变量，`y`为二分类标签。


```python
df = pd.read_csv('../data/missing_chi.csv')
df.head()
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
      <th>X_1</th>
      <th>X_2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isna().mean()
```




    X_1    0.855
    X_2    0.894
    y      0.000
    dtype: float64




```python
df.y.value_counts(normalize=True)
```




    0    0.918
    1    0.082
    Name: y, dtype: float64



事实上，有时缺失值出现或者不出现本身就是一种特征，并且在一些场合下可能与标签的正负是相关的。关于缺失出现与否和标签的正负性，在统计学中可以利用卡方检验来断言它们是否存在相关性。按照特征缺失的正例、特征缺失的负例、特征不缺失的正例、特征不缺失的负例，可以分为四种情况，设它们分别对应的样例数为$n_{11}, n_{10}, n_{01}, n_{00}$。假若它们是不相关的，那么特征缺失中正例的理论值，就应该接近于特征缺失总数$\times$总体正例的比例，即：

$$E_{11} = n_{11} \approx (n_{11}+n_{10})\times\frac{n_{11}+n_{01}}{n_{11}+n_{10}+n_{01}+n_{00}} = F_{11}$$

其他的三种情况同理。现将实际值和理论值分别记作$E_{ij}, F_{ij}$，那么希望下面的统计量越小越好，即代表实际值接近不相关情况的理论值：

$$S = \sum_{i\in \{0,1\}}\sum_{j\in \{0,1\}} \frac{(E_{ij}-F_{ij})^2}{F_{ij}}$$

可以证明上面的统计量近似服从自由度为$1$的卡方分布，即$S\overset{\cdot}{\sim} \chi^2(1)$。因此，可通过计算$P(\chi^2(1)>S)$的概率来进行相关性的判别，一般认为当此概率小于$0.05$时缺失情况与标签正负存在相关关系，即不相关条件下的理论值与实际值相差较大。

上面所说的概率即为统计学上关于$2\times2$列联表检验问题的$p$值， 它可以通过`scipy.stats.chi2(S, 1)`得到。请根据上面的材料，分别对`X_1, X_2`列进行检验。

**我的解答：**  

通过该题，可利用`pd.crosstab`方法统计元素组合[(NaN, 1), (NaN, 0), (NotNaN, 1), (NotNaN, 0)]出现的频数


```python
# 把所有为NaN的值标记为NaNValue，其他值标记为NotNaNValue
df_X1_flag = df['X_1'].fillna("NaNValue").mask(df['X_1'].notna()).fillna("NotNaNValue")
df_1 = pd.crosstab(df_X1_flag, df['y'], margins=True)
df_1
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
      <th>y</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>X_1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NaNValue</th>
      <td>785</td>
      <td>70</td>
      <td>855</td>
    </tr>
    <tr>
      <th>NotNaNValue</th>
      <td>133</td>
      <td>12</td>
      <td>145</td>
    </tr>
    <tr>
      <th>All</th>
      <td>918</td>
      <td>82</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_X2_flag = df['X_2'].fillna("NaNValue").mask(df['X_2'].notna()).fillna("NotNaNValue")
df_2 = pd.crosstab(df_X2_flag, df['y'], margins=True)
df_2
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
      <th>y</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>X_2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NaNValue</th>
      <td>894</td>
      <td>0</td>
      <td>894</td>
    </tr>
    <tr>
      <th>NotNaNValue</th>
      <td>24</td>
      <td>82</td>
      <td>106</td>
    </tr>
    <tr>
      <th>All</th>
      <td>918</td>
      <td>82</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def compute_S(df):
    res = []
    for i in [0, 1]:
        for j in [0, 1]:
            E = df.iloc[i, j]
            F = df.iloc[i, 2] * df.iloc[2, j] / df.iloc[2, 2]
            res.append((E-F)**2/F)
    return sum(res)
```


```python
from scipy.stats import chi2

chi2.sf(compute_S(df_1), 1)
```




    0.9712760884395901




```python
chi2.sf(compute_S(df_2), 1)
```




    7.459641265637543e-166



可知X1的概率大于0.05，X2的概率小于0.05，故特征X2的缺失情况与标签正负存在相关关系，不能删除，特征X1可以删除

### 3.2 Ex2：用回归模型解决分类问题

`KNN`是一种监督式学习模型，既可以解决回归问题，又可以解决分类问题。对于分类变量，利用`KNN`分类模型可以实现其缺失值的插补，思路是度量缺失样本的特征与所有其他样本特征的距离，当给定了模型参数`n_neighbors=n`时，计算离该样本距离最近的$n$个样本点中最多的那个类别，并把这个类别作为该样本的缺失预测类别，具体如下图所示，未知的类别被预测为黄色：

<img src="./pandas20/images/ch7_ex.png" width="25%">

上面有色点的特征数据提供如下：


```python
df = pd.read_excel('../data/color.xlsx')
df.head(3)
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
      <th>X1</th>
      <th>X2</th>
      <th>Color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.5</td>
      <td>2.8</td>
      <td>Blue</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.5</td>
      <td>1.8</td>
      <td>Blue</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.8</td>
      <td>2.8</td>
      <td>Blue</td>
    </tr>
  </tbody>
</table>
</div>



已知待预测的样本点为$X_1=0.8, X_2=-0.2$，那么预测类别可以如下写出：


```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(df.iloc[:,:2], df.Color)
clf.predict([[0.8, -0.2]])
```




    array(['Yellow'], dtype=object)



1. 对于回归问题而言，需要得到的是一个具体的数值，因此预测值由最近的$n$个样本对应的平均值获得。请把上面的这个分类问题转化为回归问题，仅使用`KNeighborsRegressor`来完成上述的`KNeighborsClassifier`功能。
2. 请根据第1问中的方法，对`audit`数据集中的`Employment`变量进行缺失值插补。


```python
df = pd.read_csv('../data/audit.csv')
df.head(3)
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
      <th>ID</th>
      <th>Age</th>
      <th>Employment</th>
      <th>Marital</th>
      <th>Income</th>
      <th>Gender</th>
      <th>Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1004641</td>
      <td>38</td>
      <td>Private</td>
      <td>Unmarried</td>
      <td>81838.00</td>
      <td>Female</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1010229</td>
      <td>35</td>
      <td>Private</td>
      <td>Absent</td>
      <td>72099.00</td>
      <td>Male</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1024587</td>
      <td>32</td>
      <td>Private</td>
      <td>Divorced</td>
      <td>154676.74</td>
      <td>Male</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**

**第1问：**


```python
from sklearn.neighbors import KNeighborsRegressor
```


```python
df = pd.read_excel('../data/color.xlsx')
```


```python
# 进行one-hot编码
df_color = pd.get_dummies(df['Color'])

# 得到所有类别的概率
res = []
for color in df_color.columns:
    clf = KNeighborsRegressor(n_neighbors=6)
    clf.fit(df.iloc[:,:2], df_color[color])
    predict_value = clf.predict([[0.8, -0.2]])[0]
    res.append(predict_value)
```


```python
res
```




    [0.16666666666666666, 0.3333333333333333, 0.5]




```python
# 取出概率最大的那个类别，即为所求
df_color.columns[np.array(res).argmax()]
```




    'Yellow'



**第2问：**


```python
df = pd.read_csv('../data/audit.csv')
df.head(3)
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
      <th>ID</th>
      <th>Age</th>
      <th>Employment</th>
      <th>Marital</th>
      <th>Income</th>
      <th>Gender</th>
      <th>Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1004641</td>
      <td>38</td>
      <td>Private</td>
      <td>Unmarried</td>
      <td>81838.00</td>
      <td>Female</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1010229</td>
      <td>35</td>
      <td>Private</td>
      <td>Absent</td>
      <td>72099.00</td>
      <td>Male</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1024587</td>
      <td>32</td>
      <td>Private</td>
      <td>Divorced</td>
      <td>154676.74</td>
      <td>Male</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_df = df.copy()
```


```python
# 对Age，Income，Hours进行归一化处理
df_normalize = my_df[['Age','Income','Hours']].apply(lambda x:(x-x.min())/(x.max()-x.min()))
```


```python
# 对Marital，Gender进行one-hot编码
df_one_hot = pd.get_dummies(my_df[['Marital', 'Gender']])
```


```python
# 构造类似于第1问中的数据集
my_df = pd.concat([df_normalize, df_one_hot, df['Employment']], axis=1)
```


```python
my_df.head()
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
      <th>Age</th>
      <th>Income</th>
      <th>Hours</th>
      <th>Marital_Absent</th>
      <th>Marital_Divorced</th>
      <th>Marital_Married</th>
      <th>Marital_Married-spouse-absent</th>
      <th>Marital_Unmarried</th>
      <th>Marital_Widowed</th>
      <th>Gender_Female</th>
      <th>Gender_Male</th>
      <th>Employment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.287671</td>
      <td>0.168997</td>
      <td>0.724490</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.246575</td>
      <td>0.148735</td>
      <td>0.295918</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.205479</td>
      <td>0.320539</td>
      <td>0.397959</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.383562</td>
      <td>0.056453</td>
      <td>0.551020</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.589041</td>
      <td>0.014477</td>
      <td>0.397959</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Private</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = my_df[my_df.Employment.notna()]
X_test = my_df[my_df.Employment.isna()]
```


```python
# 进行one-hot编码
df_employment = pd.get_dummies(X_train['Employment'])

# 得到所有类别的概率
res = []
for employment in df_employment.columns:
    clf = KNeighborsRegressor(n_neighbors=6)
    clf.fit(X_train.iloc[:,:-1], df_employment[employment])
    predict_value = clf.predict(X_test.iloc[:,:-1]).reshape(-1,1)
    res.append(predict_value)
```


```python
# 得到该行的最大值索引
np.hstack(res).argmax(1)
```




    array([2, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
           4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
           2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
           4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4,
           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int64)




```python
# 构造成为一个pd.Series
Y_test = pd.Series(df_employment.columns[pd.Series(np.hstack(res).argmax(1))].values)
```


```python
# 将Employment为nan的值进行赋值
df.loc[df.Employment.isna(), 'Employment'] = Y_test.values
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 7 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   ID          2000 non-null   int64  
     1   Age         2000 non-null   int64  
     2   Employment  2000 non-null   object 
     3   Marital     2000 non-null   object 
     4   Income      2000 non-null   float64
     5   Gender      2000 non-null   object 
     6   Hours       2000 non-null   int64  
    dtypes: float64(1), int64(3), object(3)
    memory usage: 109.5+ KB
    
