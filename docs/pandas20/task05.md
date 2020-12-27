# Task5 变形

## 1 知识梳理（重点记忆）


```python
import pandas as pd
import numpy as np
```

## 2 练一练

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
      <td>0:04:34</td>
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
      <td>0:04:20</td>
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
      <td>0:05:22</td>
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
      <td>0:04:08</td>
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
      <td>0:05:22</td>
    </tr>
  </tbody>
</table>
</div>



1. `mean`聚合函数，统计身高`Height`的均值


```python
%timeit -n 30 pd.crosstab(index = df.School, columns = df.Transfer, values = df.Height, aggfunc = 'mean')
```

    30 loops, best of 5: 7.32 ms per loop
    


```python
%timeit -n 30 df.pivot_table(index = 'School', columns = 'Transfer', values = 'Height', aggfunc = 'mean')
```

    30 loops, best of 5: 6.76 ms per loop
    

2. `max`聚合函数，统计体重`Weight`的最大值


```python
%timeit -n 30 pd.crosstab(index = df.School, columns = df.Transfer, values = df.Weight, aggfunc = 'max')
```

    30 loops, best of 5: 7.2 ms per loop
    


```python
%timeit -n 30 df.pivot_table(index = 'School', columns = 'Transfer', values = 'Weight', aggfunc = 'max')
```

    30 loops, best of 5: 6.68 ms per loop
    

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

<img src="../source/_static/Ex5_1.png" width="35%">

2. 将第1问中的结果恢复为原表。
3. 按`State`分别统计每年的报告数量总和，其中`State, YYYY`分别为列索引和行索引，要求分别使用`pivot_table`函数与`groupby+unstack`两种不同的策略实现，并体会它们之间的联系。

**我的解答：**  

**第1问：**


```python
res = df.pivot(index=['State', 'COUNTY', 'SubstanceName'], columns='YYYY', values='DrugReports')
res = res.reset_index().rename_axis(columns={'YYYY':''})
res.head()
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



**第2问：**

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


