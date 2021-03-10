# Task4 分组 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 分组模式及其对象
- 每个组可以返回一个标量值
- 每个组可以返回一个Series类型
- 可以返回整个组所在行，该类型是`DataFrame`类型

### 1.2 agg方法
- 将多个聚合函数，采用列表的形式传参
- 使用字典的方式，针对不同的列进行聚合统计
- 使用自定义函数进行计算，迭代值是每列
- 通过元组进行索引列的重命名

### 1.3 变换和过滤
#### 1.3.1 transform方法
- 使用group对象，迭代值是每列
- 可传入聚合方法对应的字符串

### 1.3.2 组索引与过滤
使用`filter`方法进行组的筛选，迭代值是每个分组的`DataFrame`对象

### 1.4 跨组分组
使用`apply`方法，传入的函数只允许返回布尔值，即条件函数

## 2 练一练

### 2.1 第1题

请根据上下四分位数分割，将体重分为high、normal、low三组，统计身高的均值。

**我的解答：**


```python
import pandas as pd

df = pd.read_csv('../data/learn_pandas.csv')
```


```python
low_condition = df.Weight < df.Weight.quantile(0.25)
df.groupby(low_condition)['Height'].mean()
```




    Weight
    False    165.950704
    True     153.753659
    Name: Height, dtype: float64




```python
normal_condition = (df.Weight.quantile(0.25) < df.Weight) & (df.Weight < df.Weight.quantile(0.75))
df.groupby(normal_condition)['Height'].mean()
```




    Weight
    False    164.084000
    True     162.174699
    Name: Height, dtype: float64




```python
high_condition = df.Weight > df.Weight.quantile(0.75)
df.groupby(high_condition)['Height'].mean()
```




    Weight
    False    159.727660
    True     174.935714
    Name: Height, dtype: float64



### 2.2 第2题
上一小节介绍了可以通过`drop_duplicates`得到具体的组类别，现请用`groups`属性完成类似的功能。


```python
df[['School', 'Gender']].drop_duplicates()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peking University</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fudan University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fudan University</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tsinghua University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Peking University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Tsinghua University</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
groups = df.groupby(['School', 'Gender']).groups
school_list, gender_list, index_list = [], [], []
for key, value in groups.items():
    index_list.append(value[0])
    school_list.append(key[0])
    gender_list.append(key[1])

pd.DataFrame(data={'School':school_list, 'Gender':gender_list}, index=index_list).sort_index()  
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peking University</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fudan University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fudan University</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tsinghua University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Peking University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Tsinghua University</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3 第3题
请查阅文档，明确`all/any/mad/skew/sem/prod`函数的含义。

**我的解答：**


```python
import numpy as np

np.random.seed(0)
df_ex3 = pd.DataFrame(np.random.randint(0,2,(6,4)), columns=['a', 'b', 'c', 'd'])
df_ex3['e'] = [[i, j] for (i, j) in zip(df_ex3['a'], df_ex3['b'])]
df_ex3
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>[1, 1]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[1, 1]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>[0, 0]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>[0, 1]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果分组中的该列的所有值（1为True，0为False）都是True，则为True，否则为False
df_ex3.groupby(['a', 'b']).all()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
    <tr>
      <th>a</th>
      <th>b</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果分组中的该列的任何值为（1为True，0为False）是True，则为True，否则为False
df_ex3.groupby(['a', 'b']).any()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
    <tr>
      <th>a</th>
      <th>b</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <th>1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 计算各分组的平均绝对偏差
df_ex3.groupby(['a', 'b']).mad()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>a</th>
      <th>b</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.444444</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <th>1</th>
      <td>0.000000</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 计算各分组的偏度
df_ex3.groupby(['a', 'b']).skew()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>a</th>
      <th>b</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.732051</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 计算各分组的平均无偏标准误差
df_ex3.groupby(['a', 'b']).sem()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>a</th>
      <th>b</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.333333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <th>1</th>
      <td>0.000000</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 计算各分组的乘积
df_ex3.groupby(['a', 'b']).prod()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>a</th>
      <th>b</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.4 第4题
请使用【b】中的传入字典的方法完成【a】中等价的聚合任务


```python
df = pd.read_csv('../data/learn_pandas.csv')
gb = df.groupby('Gender')[['Height', 'Weight']]

# 示例【a】
gb.agg(['sum', 'idxmax', 'skew'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Height</th>
      <th colspan="3" halign="left">Weight</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>idxmax</th>
      <th>skew</th>
      <th>sum</th>
      <th>idxmax</th>
      <th>skew</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>21014.0</td>
      <td>28</td>
      <td>-0.219253</td>
      <td>6469.0</td>
      <td>28</td>
      <td>-0.268482</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>8854.9</td>
      <td>193</td>
      <td>0.437535</td>
      <td>3929.0</td>
      <td>2</td>
      <td>-0.332393</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
methods = ['sum','idxmax', 'skew']

gb.agg({'Height':methods, 'Weight':methods})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Height</th>
      <th colspan="3" halign="left">Weight</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>idxmax</th>
      <th>skew</th>
      <th>sum</th>
      <th>idxmax</th>
      <th>skew</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>21014.0</td>
      <td>28</td>
      <td>-0.219253</td>
      <td>6469.0</td>
      <td>28</td>
      <td>-0.268482</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>8854.9</td>
      <td>193</td>
      <td>0.437535</td>
      <td>3929.0</td>
      <td>2</td>
      <td>-0.332393</td>
    </tr>
  </tbody>
</table>
</div>



### 2.5 第5题
在`groupby`对象中可以使用`describe`方法进行统计信息汇总，请同时使用多个聚合函数，完成与该方法相同的功能。


```python
df = pd.read_csv('../data/learn_pandas.csv')
gb = df.groupby('Gender')[['Height', 'Weight']]

gb.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Height</th>
      <th colspan="8" halign="left">Weight</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>132.0</td>
      <td>159.19697</td>
      <td>5.053982</td>
      <td>145.4</td>
      <td>155.675</td>
      <td>159.6</td>
      <td>162.825</td>
      <td>170.2</td>
      <td>135.0</td>
      <td>47.918519</td>
      <td>5.405983</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>48.0</td>
      <td>52.00</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>51.0</td>
      <td>173.62549</td>
      <td>7.048485</td>
      <td>155.7</td>
      <td>168.900</td>
      <td>173.4</td>
      <td>177.150</td>
      <td>193.9</td>
      <td>54.0</td>
      <td>72.759259</td>
      <td>7.772557</td>
      <td>51.0</td>
      <td>69.0</td>
      <td>73.0</td>
      <td>78.75</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
def quantile(n):
    def quantile_(x):
        return x.quantile(n/100)
    quantile_.__name__ = '{0}%'.format(n)
    return quantile_

methods = ['count','mean', 'std', 'min', quantile(25), quantile(50), quantile(75), 'max']

res = gb.agg({'Height':methods, 'Weight':methods})
res = res.astype(np.float64)
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Height</th>
      <th colspan="8" halign="left">Weight</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>132.0</td>
      <td>159.19697</td>
      <td>5.053982</td>
      <td>145.4</td>
      <td>155.675</td>
      <td>159.6</td>
      <td>162.825</td>
      <td>170.2</td>
      <td>135.0</td>
      <td>47.918519</td>
      <td>5.405983</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>48.0</td>
      <td>52.00</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>51.0</td>
      <td>173.62549</td>
      <td>7.048485</td>
      <td>155.7</td>
      <td>168.900</td>
      <td>173.4</td>
      <td>177.150</td>
      <td>193.9</td>
      <td>54.0</td>
      <td>72.759259</td>
      <td>7.772557</td>
      <td>51.0</td>
      <td>69.0</td>
      <td>73.0</td>
      <td>78.75</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.6 第6题
在`groupby`对象中，`rank`方法也是一个实用的变换函数，请查阅它的功能并给出一个使用的例子。

**我的解答：**

`rank`方法主要功能是对其它列的属于同分组数据进行数值大小排序，重点参数如下表：  

|参数|描述|
|---|:---|
|method|如果值相同，采用方法进行计算排名，默认方法为average，可选方法为average、min、max、first、dense|
|ascending|默认升序为True，降序为False|
|na_option|NaN的数据位置，默认为keep（保持原位）,可选值为keep、top、bottom|
|pct|计算分组数据的百分比值|
|axis|默认为0，0为列，1为行|


```python
np.random.seed(0)
df_ex6 = pd.DataFrame(np.random.randint(0,6,(6,4)), columns=['a', 'b', 'c', 'd'])
df_ex6
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
gb = df_ex6.groupby('a')
```


```python
gb.rank(method='first')
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.7 第7题
对于`transform`方法无法像`agg`一样，通过传入字典来对指定列使用特定的变换，如果需要在一次`transform`的调用中实现这种功能，请给出解决方案。

**我的解答：**


```python
df = pd.read_csv('../data/learn_pandas.csv')
gb = df.groupby('Gender')[['Height', 'Weight']]
```


```python
def my_func(method_dict):
    def my_(x):
        method = method_dict[x.name]
        if method == 'standardized':
            return (x-x.mean())/x.std()
        elif method == 'max':
            return x.max()
    return my_

gb.transform(my_func({'Height':'standardized', 'Weight':'max'})).head()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.058760</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.010925</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.167063</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.053133</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.8 第8题
从概念上说，索引功能是组过滤功能的子集，请使用`filter`函数完成`loc[...]`的功能，这里假设"`...`"是元素列表。 


```python
df = pd.read_csv('../data/learn_pandas.csv', usecols = ['School', 'Grade', 'Name', 'Gender', 'Weight', 'Transfer'])
df_demo = df.set_index('Name')
df_demo.loc[['Qiang Sun','Quan Zhao'], ['School','Gender']]
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Gender</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Qiang Sun</th>
      <td>Tsinghua University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>Qiang Sun</th>
      <td>Tsinghua University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>Qiang Sun</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>Quan Zhao</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
names, cols= ['Qiang Sun','Quan Zhao'], ['School','Gender']

def my_filter(df, names, cols):
    gb = df.groupby('Name')[cols]
    name_indexs = []
    for name in names:
        name_indexs += gb.get_group(name).index.tolist()    
    
    res = gb.filter(lambda x : len(set(x.index.values.tolist()) & set(name_indexs)) > 0)
    return res
res = my_filter(df, names, cols)
res
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>Tsinghua University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Tsinghua University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



### 2.9 第9题
请尝试在`apply`传入的自定义函数中，根据组的某些特征返回相同长度但索引不同的`Series`，会报错吗？

**我的解答：**


```python
df = pd.read_csv('../data/learn_pandas.csv')
gb = df.groupby(['Gender','Test_Number'])[['Height','Weight']]
gb.count()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>Test_Number</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Female</th>
      <th>1</th>
      <td>63</td>
      <td>64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>19</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Male</th>
      <th>1</th>
      <td>27</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
def my_func_ex9(x):
    if x.shape[0] < 100:
        return pd.Series([0,0], index=['a', 'b'])
    else:
        return pd.Series([0,0], index=['b', 'c'])

try:
    gb.apply(my_func_ex9)
except Exception as e:
    Exception_Msg = e
    print("Exception Message:", Exception_Msg)    
```

执行函数之后可知，返回相同长度但索引不同的`Series`会报错，原因是`Series`的`name`必须是可哈希的（不可变的），但是代码中通过强制返回索引不同的`Series`名字，导致报错。

### 2.10 第10题
请尝试在`apply`传入的自定义函数中，根据组的某些特征返回相同大小但列索引不同的`DataFrame`，会报错吗？如果只是行索引不同，会报错吗？

**我的解答：**


```python
df = pd.read_csv('../data/learn_pandas.csv')
gb = df.groupby(['Gender'])[['Height','Weight']]
gb.count()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>132</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>51</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 返回相同大小但列索引不同的DataFrame
def my_func_ex10(x):
    if x.shape[0] > 100:
        return pd.DataFrame(np.ones((2,2)), index = ['a','b'], columns=pd.Index([('w','x'),('y','z')]))
    else:
        return pd.DataFrame(np.ones((2,2)), index = ['a','b'], columns=pd.Index([('w','d'),('y','f')]))
        
try:
    gb.apply(my_func_ex10)
except Exception as e:
    Exception_Msg = e
    print("Exception Message:", Exception_Msg)
```


```python
gb.apply(my_func_ex10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">w</th>
      <th colspan="2" halign="left">y</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>d</th>
      <th>x</th>
      <th>f</th>
      <th>z</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Female</th>
      <th>a</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Male</th>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 返回相同大小但行索引不同的DataFrame
def my_func_ex10(x):
    if x.shape[0] > 100:
        return pd.DataFrame(np.ones((2,2)), index = ['a','b'], columns=pd.Index([('w','x'),('y','z')]))
    else:
        return pd.DataFrame(np.ones((2,2)), index = ['g','h'], columns=pd.Index([('w','d'),('y','f')]))
        
try:
    gb.apply(my_func_ex10)
except Exception as e:
    Exception_Msg = e
    print("Exception Message:", Exception_Msg)
```


```python
gb.apply(my_func_ex10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">w</th>
      <th colspan="2" halign="left">y</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>d</th>
      <th>x</th>
      <th>f</th>
      <th>z</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Female</th>
      <th>a</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Male</th>
      <th>g</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>h</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



观察得知，程序执行不会报错

### 2.11 第11题
在`groupby`对象中还定义了`cov`和`corr`函数，从概念上说也属于跨列的分组处理。请利用之前定义的`gb`对象，使用apply函数实现与`gb.cov()`同样的功能并比较它们的性能。


```python
df = pd.read_csv('../data/learn_pandas.csv')
gb = df.groupby(['Gender','Test_Number'])[['Height','Weight']]
gb.cov()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>Test_Number</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">Female</th>
      <th rowspan="2" valign="top">1</th>
      <th>Height</th>
      <td>20.963600</td>
      <td>21.452034</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>21.452034</td>
      <td>26.438244</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>Height</th>
      <td>31.615680</td>
      <td>30.386170</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>30.386170</td>
      <td>34.568250</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Height</th>
      <td>23.582395</td>
      <td>20.801307</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>20.801307</td>
      <td>23.228070</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Male</th>
      <th rowspan="2" valign="top">1</th>
      <th>Height</th>
      <td>42.638234</td>
      <td>48.785833</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>48.785833</td>
      <td>67.669951</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>Height</th>
      <td>57.041732</td>
      <td>38.224183</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>38.224183</td>
      <td>37.869281</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Height</th>
      <td>56.157667</td>
      <td>84.020000</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>84.020000</td>
      <td>89.904762</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
def my_cov(x):  
    cov_height_weight = x['Height'].cov(x['Weight'])
    cov_height_height = x['Height'].cov(x['Height'])
    cov_weight_weight = x['Weight'].cov(x['Weight'])
    
    return pd.DataFrame([[cov_height_height, cov_height_weight], [cov_height_weight, cov_weight_weight]], 
                        index = ['Height','Weight'], columns = pd.Index(['Height', 'Weight']))

gb.apply(my_cov)
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>Test_Number</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">Female</th>
      <th rowspan="2" valign="top">1</th>
      <th>Height</th>
      <td>20.963600</td>
      <td>21.452034</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>21.452034</td>
      <td>26.438244</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>Height</th>
      <td>31.615680</td>
      <td>30.386170</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>30.386170</td>
      <td>34.568250</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Height</th>
      <td>23.582395</td>
      <td>20.801307</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>20.801307</td>
      <td>23.228070</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Male</th>
      <th rowspan="2" valign="top">1</th>
      <th>Height</th>
      <td>42.638234</td>
      <td>48.785833</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>48.785833</td>
      <td>67.669951</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>Height</th>
      <td>57.041732</td>
      <td>38.224183</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>38.224183</td>
      <td>37.869281</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Height</th>
      <td>56.157667</td>
      <td>84.020000</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>84.020000</td>
      <td>89.904762</td>
    </tr>
  </tbody>
</table>
</div>



性能测试对比：


```python
%timeit -n 30 gb.cov()
```

    30 loops, best of 5: 4.9 ms per loop
    


```python
%timeit -n 30 gb.apply(my_cov)
```

    30 loops, best of 5: 13.9 ms per loop
    

性能测试分析：my_cov的程序比groupby.cov()慢了一倍，应该是计算Series.cov()时间慢

## 3 练习

### 3.1 Ex1：汽车数据集
现有一份汽车数据集，其中`Brand, Disp., HP`分别代表汽车品牌、发动机蓄量、发动机输出。


```python
df = pd.read_csv('../data/car.csv')
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Brand</th>
      <th>Price</th>
      <th>Country</th>
      <th>Reliability</th>
      <th>Mileage</th>
      <th>Type</th>
      <th>Weight</th>
      <th>Disp.</th>
      <th>HP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Eagle Summit 4</td>
      <td>8895</td>
      <td>USA</td>
      <td>4.0</td>
      <td>33</td>
      <td>Small</td>
      <td>2560</td>
      <td>97</td>
      <td>113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ford Escort   4</td>
      <td>7402</td>
      <td>USA</td>
      <td>2.0</td>
      <td>33</td>
      <td>Small</td>
      <td>2345</td>
      <td>114</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ford Festiva 4</td>
      <td>6319</td>
      <td>Korea</td>
      <td>4.0</td>
      <td>37</td>
      <td>Small</td>
      <td>1845</td>
      <td>81</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>



1. 先过滤出所属`Country`数超过2个的汽车，即若该汽车的`Country`在总体数据集中出现次数不超过2则剔除，再按`Country`分组计算价格均值、价格变异系数、该`Country`的汽车数量，其中变异系数的计算方法是标准差除以均值，并在结果中把变异系数重命名为`CoV`。
2. 按照表中位置的前三分之一、中间三分之一和后三分之一分组，统计`Price`的均值。
3. 对类型`Type`分组，对`Price`和`HP`分别计算最大值和最小值，结果会产生多级索引，请用下划线把多级列索引合并为单层索引。
4. 对类型`Type`分组，对`HP`进行组内的`min-max`归一化。
5. 对类型`Type`分组，计算`Disp.`与`HP`的相关系数。

**我的解答：**

**第1问：**


```python
# 过滤出所属Country数超过2个的汽车
df_1 = df.groupby('Country').filter(lambda x: x.shape[0] > 2)
df_1.head()
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Brand</th>
      <th>Price</th>
      <th>Country</th>
      <th>Reliability</th>
      <th>Mileage</th>
      <th>Type</th>
      <th>Weight</th>
      <th>Disp.</th>
      <th>HP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Eagle Summit 4</td>
      <td>8895</td>
      <td>USA</td>
      <td>4.0</td>
      <td>33</td>
      <td>Small</td>
      <td>2560</td>
      <td>97</td>
      <td>113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ford Escort   4</td>
      <td>7402</td>
      <td>USA</td>
      <td>2.0</td>
      <td>33</td>
      <td>Small</td>
      <td>2345</td>
      <td>114</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ford Festiva 4</td>
      <td>6319</td>
      <td>Korea</td>
      <td>4.0</td>
      <td>37</td>
      <td>Small</td>
      <td>1845</td>
      <td>81</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Honda Civic 4</td>
      <td>6635</td>
      <td>Japan/USA</td>
      <td>5.0</td>
      <td>32</td>
      <td>Small</td>
      <td>2260</td>
      <td>91</td>
      <td>92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mazda Protege 4</td>
      <td>6599</td>
      <td>Japan</td>
      <td>5.0</td>
      <td>32</td>
      <td>Small</td>
      <td>2440</td>
      <td>113</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_1.groupby('Country')['Price'].agg(['mean', ('CoV', lambda x : x.std()/x.mean()), 'count'])
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>CoV</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Japan</th>
      <td>13938.052632</td>
      <td>0.387429</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Japan/USA</th>
      <td>10067.571429</td>
      <td>0.240040</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Korea</th>
      <td>7857.333333</td>
      <td>0.243435</td>
      <td>3</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>12543.269231</td>
      <td>0.203344</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>



**第2问：**


```python
df_size = df.shape[0]
condition = ['head'] * int(df_size / 3) + ['mid'] * int(df_size / 3) + ['tail'] * int(df_size / 3)

df.groupby(condition)['Price'].mean()
```




    head     9069.95
    mid     13356.40
    tail    15420.65
    Name: Price, dtype: float64



**第3问：**


```python
df_2 = df.groupby('Type').agg({'Price':['max'], 'HP':['min']})
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

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="0 class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Price</th>
      <th>HP</th>
    </tr>
    <tr>
      <th></th>
      <th>max</th>
      <th>min</th>
    </tr>
    <tr>
      <th>Type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Compact</th>
      <td>18900</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Large</th>
      <td>17257</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Medium</th>
      <td>24760</td>
      <td>110</td>
    </tr>
    <tr>
      <th>Small</th>
      <td>9995</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Sporty</th>
      <td>13945</td>
      <td>92</td>
    </tr>
    <tr>
      <th>Van</th>
      <td>15395</td>
      <td>106</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2.columns = df_2.columns.map(lambda x : '_'.join(x))
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price_max</th>
      <th>HP_min</th>
    </tr>
    <tr>
      <th>Type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Compact</th>
      <td>18900</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Large</th>
      <td>17257</td>
      <td>150</td>
    </tr>
    <tr>
      <th>Medium</th>
      <td>24760</td>
      <td>110</td>
    </tr>
    <tr>
      <th>Small</th>
      <td>9995</td>
      <td>63</td>
    </tr>
    <tr>
      <th>Sporty</th>
      <td>13945</td>
      <td>92</td>
    </tr>
    <tr>
      <th>Van</th>
      <td>15395</td>
      <td>106</td>
    </tr>
  </tbody>
</table>
</div>



**第4问：对类型`Type`分组，对HP进行组内的`min-max`归一化。**


```python
df.groupby('Type')['HP'].transform(lambda x: (x-x.min())/(x.max() - x.min())).head()
```




    0    1.00
    1    0.54
    2    0.00
    3    0.58
    4    0.80
    Name: HP, dtype: float64



**第5问：对类型Type分组，计算`Disp.`与`HP`的相关系数**


```python
gb = df.groupby('Type')[['Disp.', 'HP']]

gb.apply(lambda x: np.corrcoef(x['Disp.'], x['HP'])[0,1])
```




    Type
    Compact    0.586087
    Large     -0.242765
    Medium     0.370491
    Small      0.603916
    Sporty     0.871426
    Van        0.819881
    dtype: float64



### 3.2 Ex2：实现transform函数
* `groupby`对象的构造方法是`my_groupby(df, group_cols)`
* 支持单列分组与多列分组
* 支持带有标量广播的`my_groupby(df)[col].transform(my_func)`功能
* `pandas`的`transform`不能跨列计算，请支持此功能，即仍返回`Series`但`col`参数为多列
* 无需考虑性能与异常处理，只需实现上述功能，在给出测试样例的同时与`pandas`中的`transform`对比结果是否一致

**解题思路：**


```python
class my_groupby:
    def __init__(self, df: pd.DataFrame, group_cols):
        # 原始数据集，进行复制
        self._df = df.copy()
        # 得到分组类别，返回Series类型
        self._groups = self._df[group_cols].drop_duplicates()
        # 如果取出为Series，需要转换为DataFrame
        if isinstance(self._groups, pd.Series):
            self._group_category_df = self._groups.to_frame()
        else:
            self._group_category_df = self._groups.copy()

    def __getitem__(self, col):
        # 由于要满足[col]/[[col1, col2, ...]]，故需要getitem方法
        # 为保证head()方法的使用，需要返回DataFrame或Series类型
        self._pr_col = [col] if isinstance(col, str) else list(col)
        return self

    def transform(self, my_func) -> pd.Series:
        # 定义两个空数组，用于存储索引和数据
        index_array = np.array([])
        value_array = np.array([])
        for group_df in self.__iter_group():
            # 进行分组遍历
            if self._pr_col:
                group_df = group_df[self._pr_col]
            if group_df.shape[1] == 1:
                group_df = group_df.iloc[:, 0]
            # 执行自定义函数
            group_res = my_func(group_df)
            # 转换为Series，用于进行数据拼接
            if not isinstance(group_res, pd.Series):
                group_res = pd.Series(group_res, index=group_df.index, name=group_df.name)

            # 存储索引和数据
            index_array = np.r_[index_array, group_res.index]
            value_array = np.r_[value_array, group_res.values]

        # 将分组之后得到的数据，再进行重排，按照正常索引之后的数据
        values = pd.Series(data=value_array, index=index_array).sort_index().values
        # 结合原始数据集的索引，构建Series
        result_series = pd.Series(data=values, index=self._df.reset_index().index, name=my_func.__name__)
        return result_series

    def head(self, n=5) -> pd.DataFrame:
        '''
        该功能是每个分组取前n个
        :param n:
        :return:
        '''
        # 取每个分组的前n个数据
        res_df = pd.DataFrame()
        for group_df in self.__iter_group():
            res_df = pd.concat([res_df, group_df.head(n)], ignore_index=True)

        # 删除索引列，重新指定索引数据
        index_values = res_df['index'].values
        res_df = res_df.drop(axis=0, columns=['index'])
        res_df.index = index_values

        if self._pr_col:
            return res_df[self._pr_col]
        return res_df

    def __iter_group(self):
        # 进行分组
        for index, groups in self._group_category_df.iterrows():
            group_df = self._df.reset_index().copy()
            for col_name, target in groups.to_dict().items():
                group_df = group_df[group_df[col_name] == target]
            yield group_df
```


```python
df = pd.read_csv('../data/car.csv')
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
<table border="0 class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Brand</th>
      <th>Price</th>
      <th>Country</th>
      <th>Reliability</th>
      <th>Mileage</th>
      <th>Type</th>
      <th>Weight</th>
      <th>Disp.</th>
      <th>HP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Eagle Summit 4</td>
      <td>8895</td>
      <td>USA</td>
      <td>4.0</td>
      <td>33</td>
      <td>Small</td>
      <td>2560</td>
      <td>97</td>
      <td>113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ford Escort   4</td>
      <td>7402</td>
      <td>USA</td>
      <td>2.0</td>
      <td>33</td>
      <td>Small</td>
      <td>2345</td>
      <td>114</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ford Festiva 4</td>
      <td>6319</td>
      <td>Korea</td>
      <td>4.0</td>
      <td>37</td>
      <td>Small</td>
      <td>1845</td>
      <td>81</td>
      <td>63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Honda Civic 4</td>
      <td>6635</td>
      <td>Japan/USA</td>
      <td>5.0</td>
      <td>32</td>
      <td>Small</td>
      <td>2260</td>
      <td>91</td>
      <td>92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mazda Protege 4</td>
      <td>6599</td>
      <td>Japan</td>
      <td>5.0</td>
      <td>32</td>
      <td>Small</td>
      <td>2440</td>
      <td>113</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
def test1():
    # head()方法测试
    res = my_groupby(df, ['Type', 'Country'])['Disp.', 'HP'].head(1)
    res_df = df.groupby(['Type', 'Country'])[['Disp.', 'HP']].head(1)
    assert res.equals(res_df)


def f(s):
    # 自定义函数
    res = (s - s.min()) / (s.max() - s.min())
    return res


def test2():
    # 单列分组
    res = my_groupby(df, 'Type')['Price'].transform(f).head()
    res_df = df.groupby('Type')['Price'].transform(f).head()
    assert res.equals(res_df)


def test3():
    # 多列分组
    res = my_groupby(df, ['Type', 'Country'])['Price'].transform(f).head()
    res_df = df.groupby(['Type', 'Country'])['Price'].transform(f).head()
    assert res.equals(res_df)


def test4():
    # 标量广播
    res = my_groupby(df, 'Type')['Price'].transform(lambda x: x.mean()).head()
    res_df = df.groupby('Type')['Price'].transform(lambda x: x.mean()).head()
    assert res.equals(res_df)


def test5():
    # 跨列计算
    res = my_groupby(df, 'Type')[['Disp.', 'HP']].transform(lambda x: x['Disp.'] / x.HP).head()
    print("\n跨列计算：\n", res)
```


```python
test1()
test2()
test3()
test4()
test5()
```

    
    跨列计算：
     0    0.858407
    1    1.266667
    2    1.285714
    3    0.989130
    4    1.097087
    Name: <lambda>, dtype: float64
    
