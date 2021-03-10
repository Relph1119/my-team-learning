# Task5 变形 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 长宽表变形
- pivot函数，使用场景：长表变宽表
- pivot_table函数，使用场景：长表变宽表的同时，进行数据聚合  
参数说明：
|参数|描述|
|---|---|
|index|宽表的索引列|
|columns|宽表的列名|
|values|数据集，即资源|
|margins|bool类型，边际汇总|

- melt函数，使用场景：宽表变长表  
参数说明：
|参数|描述|
|---|---|
|id_vars|分组列|
|value_vars|长表中的类别|
|var_name|类别名|
|value_name|数据集，即资源|

- wide_to_long函数，使用场景：列名拆分，即列中包含交叉类别
参数说明：
|参数|描述|
|---|---|
|stubnames|分隔列名之后的前缀列名|
|i|索引列|
|j|分隔列名之后的后缀分组列名|
|sep|列名分隔符|
|suffix|正则表达式|

### 1.2 索引的变形
- stack函数，使用场景：把列索引的层压入行索引
- unstack函数，使用场景：把行索引转为列索引

### 1.3 其他变形函数
- crosstab函数，使用场景：在默认状态下，可以统计元素组合出现的频数
- explode函数，使用场景：对某一列的元素进行纵向的展开，其类型必须为`list`, `tuple`, `Series`, `np.ndarray`中的一种

## 2 练一练


```python
import pandas as pd
import numpy as np
```

### 2.1 第1题
在上面的边际汇总例子中，行或列的汇总为新表中行元素或者列元素的平均值，而总体的汇总为新表中四个元素的平均值。这种关系一定成立吗？若不成立，请给出一个例子来说明。

**我的解答：**


```python
df = pd.DataFrame({'Name':['San Zhang', 'San Zhang', 
                              'San Zhang', 'San Zhang',
                              'Si Li', 'Si Li', 'Si Li', 'Si Li'],
                   'Subject':['Chinese', 'Chinese', 'Math', 'Math',
                                 'Chinese', 'Chinese', 'Math', 'Math'],
                   'Grade':[80, 90, 100, 90, 70, 80, 85, 95]})
df
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
      <th>Name</th>
      <th>Subject</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Zhang</td>
      <td>Chinese</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>San Zhang</td>
      <td>Chinese</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>San Zhang</td>
      <td>Math</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>San Zhang</td>
      <td>Math</td>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Si Li</td>
      <td>Chinese</td>
      <td>70</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Si Li</td>
      <td>Chinese</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Si Li</td>
      <td>Math</td>
      <td>85</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Si Li</td>
      <td>Math</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pivot_table(index = 'Name',
               columns = 'Subject',
               values = 'Grade',
               aggfunc=lambda x:x.min(),
               margins=True)
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
      <th>Subject</th>
      <th>Chinese</th>
      <th>Math</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>San Zhang</th>
      <td>80</td>
      <td>90</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Si Li</th>
      <td>70</td>
      <td>85</td>
      <td>70</td>
    </tr>
    <tr>
      <th>All</th>
      <td>70</td>
      <td>85</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>



上面的说法不正确，总体的汇总为应该为所有Grade的值进行聚合，而不是新表中四个元素，通过查看pandas源码，`pivot.py`下的`_compute_grand_margin`函数，描述了整个计算过程：  
- 遍历data[values]
- 根据给定的函数，计算grand_margin[k] = aggfunc(v)
- 最后返回计算得到的grand_margin

```python
def _compute_grand_margin(data, values, aggfunc, margins_name: str = "All"):
    if values:
        grand_margin = {}
        for k, v in data[values].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)()
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])()
                    else:
                        grand_margin[k] = aggfunc[k](v)
                else:
                    grand_margin[k] = aggfunc(v)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: aggfunc(data.index)}
```

### 2.2 第2题
前面提到了`crosstab`的性能劣于`pivot_table`，请选用多个聚合方法进行验证。

**我的解答：**


```python
df = pd.read_csv('../data/learn_pandas.csv')
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
      <th>School</th>
      <th>Grade</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Transfer</th>
      <th>Test_Number</th>
      <th>Test_Date</th>
      <th>Time_Record</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Freshman</td>
      <td>Gaopeng Yang</td>
      <td>Female</td>
      <td>158.9</td>
      <td>46.0</td>
      <td>N</td>
      <td>1</td>
      <td>2019/10/5</td>
      <td>0&#58;04&#58;34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peking University</td>
      <td>Freshman</td>
      <td>Changqiang You</td>
      <td>Male</td>
      <td>166.5</td>
      <td>70.0</td>
      <td>N</td>
      <td>1</td>
      <td>2019/9/4</td>
      <td>0&#58;04&#58;20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Senior</td>
      <td>Mei Sun</td>
      <td>Male</td>
      <td>188.9</td>
      <td>89.0</td>
      <td>N</td>
      <td>2</td>
      <td>2019/9/12</td>
      <td>0&#58;05&#58;22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fudan University</td>
      <td>Sophomore</td>
      <td>Xiaojuan Sun</td>
      <td>Female</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>N</td>
      <td>2</td>
      <td>2020/1/3</td>
      <td>0&#58;04&#58;08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fudan University</td>
      <td>Sophomore</td>
      <td>Gaojuan You</td>
      <td>Male</td>
      <td>174.0</td>
      <td>74.0</td>
      <td>N</td>
      <td>2</td>
      <td>2019/11/6</td>
      <td>0&#58;05&#58;22</td>
    </tr>
  </tbody>
</table>
</div>



1. `mean`聚合函数，统计身高`Height`的均值


```python
%timeit -n 30 pd.crosstab(index = df.School, columns = df.Transfer, values = df.Height, aggfunc = 'mean')
```

    30 loops, best of 5: 9.62 ms per loop
    


```python
%timeit -n 30 df.pivot_table(index = 'School', columns = 'Transfer', values = 'Height', aggfunc = 'mean')
```

    30 loops, best of 5: 9.08 ms per loop
    

2. `max`聚合函数，统计体重`Weight`的最大值


```python
%timeit -n 30 pd.crosstab(index = df.School, columns = df.Transfer, values = df.Weight, aggfunc = 'max')
```

    30 loops, best of 5: 11.1 ms per loop
    


```python
%timeit -n 30 df.pivot_table(index = 'School', columns = 'Transfer', values = 'Weight', aggfunc = 'max')
```

    30 loops, best of 5: 10.2 ms per loop
    

## 3 练习
### 3.1 Ex1：美国非法药物数据集

现有一份关于美国非法药物的数据集，其中`SubstanceName, DrugReports`分别指药物名称和报告数量：


```python
df = pd.read_csv('../data/drugs.csv').sort_values(['State','COUNTY','SubstanceName'],ignore_index=True)
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
      <th>YYYY</th>
      <th>State</th>
      <th>COUNTY</th>
      <th>SubstanceName</th>
      <th>DrugReports</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011</td>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013</td>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



1. 将数据转为如下的形式：

    <img src="./pandas20/images/Ex5_1.png" width="35%">

2. 将第1问中的结果恢复为原表。
3. 按`State`分别统计每年的报告数量总和，其中`State, YYYY`分别为列索引和行索引，要求分别使用`pivot_table`函数与`groupby+unstack`两种不同的策略实现，并体会它们之间的联系。

**我的解答：**  

**第1问：**


```python
df_pivot = df.pivot(index=['State', 'COUNTY', 'SubstanceName'], columns='YYYY', values='DrugReports')
df_pivot = df_pivot.reset_index().rename_axis(columns={'YYYY':''})
df_pivot.head()
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
      <th>State</th>
      <th>COUNTY</th>
      <th>SubstanceName</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>27.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Codeine</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Fentanyl</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Heroin</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Hydrocodone</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



**第2问：** 使用`melt`进行恢复


```python
df_melt = df_pivot.melt(id_vars=['State', 'COUNTY', 'SubstanceName'],
              value_vars=df_pivot.columns[3:],
              var_name='YYYY',
              value_name='DrugReports')
df_melt.head(3)
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
      <th>State</th>
      <th>COUNTY</th>
      <th>SubstanceName</th>
      <th>YYYY</th>
      <th>DrugReports</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Codeine</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Fentanyl</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
      <th>YYYY</th>
      <th>State</th>
      <th>COUNTY</th>
      <th>SubstanceName</th>
      <th>DrugReports</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011</td>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012</td>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013</td>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Buprenorphine</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_melt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 49712 entries, 0 to 49711
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   State          49712 non-null  object 
     1   COUNTY         49712 non-null  object 
     2   SubstanceName  49712 non-null  object 
     3   YYYY           49712 non-null  object 
     4   DrugReports    24062 non-null  float64
    dtypes: float64(1), object(4)
    memory usage: 1.9+ MB
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24062 entries, 0 to 24061
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   YYYY           24062 non-null  int64 
     1   State          24062 non-null  object
     2   COUNTY         24062 non-null  object
     3   SubstanceName  24062 non-null  object
     4   DrugReports    24062 non-null  int64 
    dtypes: int64(2), object(3)
    memory usage: 940.0+ KB
    

观察可知，`df_melt`需要再进行如下操作，才能和`df`一致：
- 将`DrugReports`列为`NaN`的行删除
- 重新将列进行重排，然后将`YYYY`的类型修改为`int64`，将`DrugReports`的类型修改为`int64`


```python
df_melt.dropna(subset=['DrugReports'], inplace=True)
df_melt.head()
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
      <th>State</th>
      <th>COUNTY</th>
      <th>SubstanceName</th>
      <th>YYYY</th>
      <th>DrugReports</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Hydrocodone</td>
      <td>2010</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KY</td>
      <td>ADAIR</td>
      <td>Methadone</td>
      <td>2010</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KY</td>
      <td>ALLEN</td>
      <td>Hydrocodone</td>
      <td>2010</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>KY</td>
      <td>ALLEN</td>
      <td>Methadone</td>
      <td>2010</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>KY</td>
      <td>ALLEN</td>
      <td>Oxycodone</td>
      <td>2010</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_melt = df_melt[df.columns].sort_values(['State', 'COUNTY', 'SubstanceName'], ignore_index=True)
df_melt = df_melt.astype({'YYYY':'int64', 'DrugReports':'int64'})
```


```python
df_melt.equals(df)
```




    True



**第3问：** 按State分别统计每年的报告数量总和，其中State, YYYY分别为列索引和行索引，要求分别使用pivot_table函数与groupby+unstack两种不同的策略实现，并体会它们之间的联系。  
1. 使用`pivot_table`函数


```python
df.pivot_table(index='YYYY', columns='State', values='DrugReports', aggfunc='sum')
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
      <th>State</th>
      <th>KY</th>
      <th>OH</th>
      <th>PA</th>
      <th>VA</th>
      <th>WV</th>
    </tr>
    <tr>
      <th>YYYY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <td>10453</td>
      <td>19707</td>
      <td>19814</td>
      <td>8685</td>
      <td>2890</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>10289</td>
      <td>20330</td>
      <td>19987</td>
      <td>6749</td>
      <td>3271</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>10722</td>
      <td>23145</td>
      <td>19959</td>
      <td>7831</td>
      <td>3376</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>11148</td>
      <td>26846</td>
      <td>20409</td>
      <td>11675</td>
      <td>4046</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>11081</td>
      <td>30860</td>
      <td>24904</td>
      <td>9037</td>
      <td>3280</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>9865</td>
      <td>37127</td>
      <td>25651</td>
      <td>8810</td>
      <td>2571</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>9093</td>
      <td>42470</td>
      <td>26164</td>
      <td>10195</td>
      <td>2548</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>9394</td>
      <td>46104</td>
      <td>27894</td>
      <td>10448</td>
      <td>1614</td>
    </tr>
  </tbody>
</table>
</div>



2. 使用`groupby`和`unstack`方法


```python
df_ex1_3 = df.groupby(['State', 'YYYY'])['DrugReports'].sum().to_frame()
df_ex1_3.head(5)
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
      <th></th>
      <th>DrugReports</th>
    </tr>
    <tr>
      <th>State</th>
      <th>YYYY</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">KY</th>
      <th>2010</th>
      <td>10453</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>10289</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>10722</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>11148</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>11081</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ex1_3 = df_ex1_3.unstack(0)
df_ex1_3
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
<table border="0" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">DrugReports</th>
    </tr>
    <tr>
      <th>State</th>
      <th>KY</th>
      <th>OH</th>
      <th>PA</th>
      <th>VA</th>
      <th>WV</th>
    </tr>
    <tr>
      <th>YYYY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <td>10453</td>
      <td>19707</td>
      <td>19814</td>
      <td>8685</td>
      <td>2890</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>10289</td>
      <td>20330</td>
      <td>19987</td>
      <td>6749</td>
      <td>3271</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>10722</td>
      <td>23145</td>
      <td>19959</td>
      <td>7831</td>
      <td>3376</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>11148</td>
      <td>26846</td>
      <td>20409</td>
      <td>11675</td>
      <td>4046</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>11081</td>
      <td>30860</td>
      <td>24904</td>
      <td>9037</td>
      <td>3280</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>9865</td>
      <td>37127</td>
      <td>25651</td>
      <td>8810</td>
      <td>2571</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>9093</td>
      <td>42470</td>
      <td>26164</td>
      <td>10195</td>
      <td>2548</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>9394</td>
      <td>46104</td>
      <td>27894</td>
      <td>10448</td>
      <td>1614</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 删掉DrugReports，使用droplevel方法
df_ex1_3.droplevel(level=0, axis=1)
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
      <th>State</th>
      <th>KY</th>
      <th>OH</th>
      <th>PA</th>
      <th>VA</th>
      <th>WV</th>
    </tr>
    <tr>
      <th>YYYY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <td>10453</td>
      <td>19707</td>
      <td>19814</td>
      <td>8685</td>
      <td>2890</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>10289</td>
      <td>20330</td>
      <td>19987</td>
      <td>6749</td>
      <td>3271</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>10722</td>
      <td>23145</td>
      <td>19959</td>
      <td>7831</td>
      <td>3376</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>11148</td>
      <td>26846</td>
      <td>20409</td>
      <td>11675</td>
      <td>4046</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>11081</td>
      <td>30860</td>
      <td>24904</td>
      <td>9037</td>
      <td>3280</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>9865</td>
      <td>37127</td>
      <td>25651</td>
      <td>8810</td>
      <td>2571</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>9093</td>
      <td>42470</td>
      <td>26164</td>
      <td>10195</td>
      <td>2548</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>9394</td>
      <td>46104</td>
      <td>27894</td>
      <td>10448</td>
      <td>1614</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 Ex2：特殊的wide_to_long方法

从功能上看，`melt`方法应当属于`wide_to_long`的一种特殊情况，即`stubnames`只有一类。请使用`wide_to_long`生成`melt`一节中的`df_melted`。（提示：对列名增加适当的前缀）


```python
df = pd.DataFrame({'Class':[1,2],
                   'Name':['San Zhang', 'Si Li'],
                   'Chinese':[80, 90],
                   'Math':[80, 75]})
df
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
      <th>Class</th>
      <th>Name</th>
      <th>Chinese</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Si Li</td>
      <td>90</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_melted = df.melt(id_vars = ['Class', 'Name'],
                    value_vars = ['Chinese', 'Math'],
                    var_name = 'Subject',
                    value_name = 'Grade')
df_melted
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
      <th>Class</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>Chinese</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Si Li</td>
      <td>Chinese</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>Math</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Si Li</td>
      <td>Math</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
# 对列名增加适当的前缀
df_ex2 = df.copy()
df_ex2.rename(columns={'Chinese':'my_Chinese', 'Math':'my_Math'}, inplace=True)
df_ex2.head()
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
      <th>Class</th>
      <th>Name</th>
      <th>my_Chinese</th>
      <th>my_Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Si Li</td>
      <td>90</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ex2 = pd.wide_to_long(df_ex2,
                stubnames=['my'],
                i = ['Class', 'Name'],
                j='Subject',
                sep='_',
                suffix='.+')
df_ex2 = df_ex2.reset_index()
df_ex2
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
      <th>Class</th>
      <th>Name</th>
      <th>Subject</th>
      <th>my</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>Chinese</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>Math</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Si Li</td>
      <td>Chinese</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Si Li</td>
      <td>Math</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改列名
df_ex2.rename(columns={'my':'Grade'}, inplace=True)
```


```python
# 按照Subject排序并忽略index列
df_ex2.sort_values(['Subject'], inplace=True, ignore_index=True)
df_ex2
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
      <th>Class</th>
      <th>Name</th>
      <th>Subject</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>Chinese</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Si Li</td>
      <td>Chinese</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>San Zhang</td>
      <td>Math</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Si Li</td>
      <td>Math</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ex2.equals(df_melted)
```




    True


