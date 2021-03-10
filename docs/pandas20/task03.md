# Task03 索引 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 索引器

#### 1.1.1 loc索引器
`loc`索引器，主要用于选取指定行列的数据，使用形式为`loc[*, *]`，可使用的对象为：
- 单个元素：如果返回为多个，则为Series，如果唯一，则为单个元素
- 元素列表
- 元素切片
- 布尔表达式：类似过滤器df[conditions]，可使用`|`（或）, `&`（且）,`~`（取反）
- 函数

#### 1.1.2 iloc索引器
`iloc`索引器，和`loc`索引器类似

#### 1.1.3 query方法
`query`方法和SQL类似，方法里面传入类SQL参数，便于多个复合条件的查找，表达简洁

### 1.2 多级索引

#### 1.2.1 多级索引及其表结构
通过`.index.get_level_values(x)`方法获得索引的属性值，然后调用`.tolist()`方法可将其转换为列表


```python
import pandas as pd
import numpy as np

np.random.seed(0)
L1,L2 = ['A','B','C'],['a','b']
mul_index1 = pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
L3,L4 = ['D','E'],['d','e','f']
mul_index2 = pd.MultiIndex.from_product([L3,L4],names=('Big', 'Small'))
df = pd.DataFrame(np.random.randint(-6,7,(6,6)), index=mul_index1, columns=mul_index2)
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
      <th>Big</th>
      <th colspan="3" halign="left">D</th>
      <th colspan="3" halign="left">E</th>
    </tr>
    <tr>
      <th></th>
      <th>Small</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
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
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>6</td>
      <td>-1</td>
      <td>-6</td>
      <td>-3</td>
      <td>5</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>3</td>
      <td>-3</td>
      <td>-1</td>
      <td>-4</td>
      <td>-2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>a</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>a</th>
      <td>-1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>-2</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-6</td>
      <td>-3</td>
      <td>-1</td>
      <td>-6</td>
      <td>-4</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index.get_level_values(1).tolist()
```




    ['a', 'b', 'a', 'b', 'a', 'b']



#### 1.2.2 IndexSlice对象
通过采用`IndexSlice`对象，可以进行数据的条件选择


```python
# 选取列和大于0的数据
idx = pd.IndexSlice

df.loc[idx[:'A', lambda x:x.sum()>0]]
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
      <th>Big</th>
      <th>D</th>
      <th>E</th>
    </tr>
    <tr>
      <th></th>
      <th>Small</th>
      <th>e</th>
      <th>e</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>-1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>-4</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.2.3 多级索引的构造
- from_tuples
- from_arrays 
- from_product

### 1.3 索引的常用方法

#### 1.3.1 索引层的交换和删除


```python
# 列索引的第1层和第2层交换
df.swaplevel(1,0,axis=1).head() 
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
      <th>Small</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
    <tr>
      <th></th>
      <th>Big</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
      <th>E</th>
      <th>E</th>
      <th>E</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
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
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>6</td>
      <td>-1</td>
      <td>-6</td>
      <td>-3</td>
      <td>5</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>3</td>
      <td>-3</td>
      <td>-1</td>
      <td>-4</td>
      <td>-2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>a</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>C</th>
      <th>a</th>
      <td>-1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>-2</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 列表数字指代原来索引中的层
df.reorder_levels([1,0],axis=0).head() 
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
      <th>Big</th>
      <th colspan="3" halign="left">D</th>
      <th colspan="3" halign="left">E</th>
    </tr>
    <tr>
      <th></th>
      <th>Small</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
    <tr>
      <th>Lower</th>
      <th>Upper</th>
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
      <th>a</th>
      <th>A</th>
      <td>6</td>
      <td>-1</td>
      <td>-6</td>
      <td>-3</td>
      <td>5</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>b</th>
      <th>A</th>
      <td>1</td>
      <td>3</td>
      <td>-3</td>
      <td>-1</td>
      <td>-4</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>a</th>
      <th>B</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>b</th>
      <th>B</th>
      <td>-5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>a</th>
      <th>C</th>
      <td>-1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>-2</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 删除第1层的列索引
df.droplevel(1,axis=1)
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
      <th>Big</th>
      <th>D</th>
      <th>D</th>
      <th>D</th>
      <th>E</th>
      <th>E</th>
      <th>E</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
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
      <th rowspan="2" valign="top">A</th>
      <th>a</th>
      <td>6</td>
      <td>-1</td>
      <td>-6</td>
      <td>-3</td>
      <td>5</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>3</td>
      <td>-3</td>
      <td>-1</td>
      <td>-4</td>
      <td>-2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>a</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>a</th>
      <td>-1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>-2</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-6</td>
      <td>-3</td>
      <td>-1</td>
      <td>-6</td>
      <td>-4</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.3.2 索引属性的修改
- 通过`rename_axis`可以对索引层的名字进行修改，常用的修改方式是传入字典的映射
- 通过`rename`可以对索引的值进行修改，如果是多级索引需要指定修改的层号`level`

### 1.4 索引的运算
- $S_A \cap S_B$：`S_A.intersection(S_B)`、`S_A & S_B`
- $S_A \cup S_B$：`S_A.union(S_B)`、`S_A | S_B`
- $S_A - S_B$：`S_A.difference(S_B)`、`(S_A ^ S_B) & S_A`
- $S_A\triangle S_B$：`S_A.symmetric\_difference(S_B)`、`S_A ^ S_B`

## 2 练一练

### 2.1 第1题
`select_dtypes`是一个实用函数，它能够从表中选出相应类型的列，若要选出所有数值型的列，只需使用`.select_dtypes('number')`，请利用布尔列表选择的方法结合`DataFrame`的`dtypes`属性在`learn_pandas`数据集上实现这个功能。

**我的解答：**


```python
df = pd.read_csv('../data/learn_pandas.csv', usecols = ['School', 'Grade', 'Name', 'Gender', 'Weight', 'Transfer'])
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
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Freshman</td>
      <td>Gaopeng Yang</td>
      <td>Female</td>
      <td>46.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peking University</td>
      <td>Freshman</td>
      <td>Changqiang You</td>
      <td>Male</td>
      <td>70.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Senior</td>
      <td>Mei Sun</td>
      <td>Male</td>
      <td>89.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fudan University</td>
      <td>Sophomore</td>
      <td>Xiaojuan Sun</td>
      <td>Female</td>
      <td>41.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fudan University</td>
      <td>Sophomore</td>
      <td>Gaojuan You</td>
      <td>Male</td>
      <td>74.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 6 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   School    200 non-null    object 
     1   Grade     200 non-null    object 
     2   Name      200 non-null    object 
     3   Gender    200 non-null    object 
     4   Weight    189 non-null    float64
     5   Transfer  188 non-null    object 
    dtypes: float64(1), object(5)
    memory usage: 9.5+ KB
    


```python
# 可以观察到，只有Weight列符合条件，类型为number
df.select_dtypes('number').head()
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
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 利用布尔列表选择的方法结合DataFrame的dtypes属性实现
import numpy as np

df.loc[:,df.dtypes[df.dtypes == np.number].index].head()
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
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 第2题
与单层索引类似，若存在重复元素，则不能使用切片，请去除重复索引后给出一个元素切片的例子。

**我的解答：**


```python
df_multi = df.set_index(['School', 'Grade'])
df_multi = df_multi.sort_index()
df_multi.head()
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
      <th>Name</th>
      <th>Gender</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
    <tr>
      <th>School</th>
      <th>Grade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Fudan University</th>
      <th>Freshman</th>
      <td>Changqiang Yang</td>
      <td>Female</td>
      <td>49.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <td>Gaoqiang Qin</td>
      <td>Female</td>
      <td>63.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <td>Gaofeng Zhao</td>
      <td>Female</td>
      <td>43.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <td>Yanquan Wang</td>
      <td>Female</td>
      <td>55.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <td>Feng Wang</td>
      <td>Male</td>
      <td>74.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dup = df_multi.reset_index().drop_duplicates(subset=['School','Grade'], keep='first').set_index(['School','Grade'])
df_dup
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
      <th>Name</th>
      <th>Gender</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
    <tr>
      <th>School</th>
      <th>Grade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Fudan University</th>
      <th>Freshman</th>
      <td>Changqiang Yang</td>
      <td>Female</td>
      <td>49.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Junior</th>
      <td>Yanli You</td>
      <td>Female</td>
      <td>48.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Senior</th>
      <td>Chengpeng Zheng</td>
      <td>Female</td>
      <td>38.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <td>Xiaojuan Sun</td>
      <td>Female</td>
      <td>41.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Peking University</th>
      <th>Freshman</th>
      <td>Changqiang You</td>
      <td>Male</td>
      <td>70.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Junior</th>
      <td>Juan Xu</td>
      <td>Female</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Senior</th>
      <td>Changli Lv</td>
      <td>Female</td>
      <td>41.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <td>Changmei Xu</td>
      <td>Female</td>
      <td>43.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Shanghai Jiao Tong University</th>
      <th>Freshman</th>
      <td>Gaopeng Yang</td>
      <td>Female</td>
      <td>46.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Junior</th>
      <td>Feng Zheng</td>
      <td>Female</td>
      <td>51.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Senior</th>
      <td>Mei Sun</td>
      <td>Male</td>
      <td>89.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <td>Yanfeng Qian</td>
      <td>Female</td>
      <td>48.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">Tsinghua University</th>
      <th>Freshman</th>
      <td>Xiaoli Qian</td>
      <td>Female</td>
      <td>51.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Junior</th>
      <td>Gaoqiang Qian</td>
      <td>Female</td>
      <td>50.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Senior</th>
      <td>Xiaomei Zhou</td>
      <td>Female</td>
      <td>57.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <td>Li Wang</td>
      <td>Male</td>
      <td>79.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_dup.loc[('Fudan University', 'Freshman'):('Peking University', 'Junior')]
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
      <th>Name</th>
      <th>Gender</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
    <tr>
      <th>School</th>
      <th>Grade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Fudan University</th>
      <th>Freshman</th>
      <td>Changqiang Yang</td>
      <td>Female</td>
      <td>49.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Junior</th>
      <td>Yanli You</td>
      <td>Female</td>
      <td>48.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Senior</th>
      <td>Chengpeng Zheng</td>
      <td>Female</td>
      <td>38.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <td>Xiaojuan Sun</td>
      <td>Female</td>
      <td>41.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Peking University</th>
      <th>Freshman</th>
      <td>Changqiang You</td>
      <td>Male</td>
      <td>70.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>Junior</th>
      <td>Juan Xu</td>
      <td>Female</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3 第3题
尝试在`rename_axis`中使用函数完成与例子中一样的功能。


```python
np.random.seed(0)
L1,L2,L3 = ['A','B'],['a','b'],['alpha','beta']
mul_index1 = pd.MultiIndex.from_product([L1,L2,L3], names=('Upper', 'Lower','Extra'))
L4,L5,L6 = ['C','D'],['c','d'],['cat','dog']
mul_index2 = pd.MultiIndex.from_product([L4,L5,L6], names=('Big', 'Small', 'Other'))
df_ex = pd.DataFrame(np.random.randint(-9,10,(8,8)), index=mul_index1,  columns=mul_index2)
df_ex
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
      <th></th>
      <th>Big</th>
      <th colspan="4" halign="left">C</th>
      <th colspan="4" halign="left">D</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Small</th>
      <th colspan="2" halign="left">c</th>
      <th colspan="2" halign="left">d</th>
      <th colspan="2" halign="left">c</th>
      <th colspan="2" halign="left">d</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Other</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
    </tr>
    <tr>
      <th>Upper</th>
      <th>Lower</th>
      <th>Extra</th>
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
      <th rowspan="4" valign="top">A</th>
      <th rowspan="2" valign="top">a</th>
      <th>alpha</th>
      <td>3</td>
      <td>6</td>
      <td>-9</td>
      <td>-6</td>
      <td>-6</td>
      <td>-2</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-5</td>
      <td>-3</td>
      <td>3</td>
      <td>-8</td>
      <td>-3</td>
      <td>-2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>alpha</th>
      <td>-4</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>7</td>
      <td>-4</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-9</td>
      <td>9</td>
      <td>-6</td>
      <td>8</td>
      <td>5</td>
      <td>-2</td>
      <td>-9</td>
      <td>-8</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">B</th>
      <th rowspan="2" valign="top">a</th>
      <th>alpha</th>
      <td>0</td>
      <td>-9</td>
      <td>1</td>
      <td>-6</td>
      <td>2</td>
      <td>9</td>
      <td>-7</td>
      <td>-9</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-9</td>
      <td>-5</td>
      <td>-4</td>
      <td>-3</td>
      <td>-1</td>
      <td>8</td>
      <td>6</td>
      <td>-5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>alpha</th>
      <td>0</td>
      <td>1</td>
      <td>-8</td>
      <td>-8</td>
      <td>-2</td>
      <td>0</td>
      <td>-6</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>2</td>
      <td>5</td>
      <td>9</td>
      <td>-9</td>
      <td>5</td>
      <td>-6</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**

原功能：


```python
df_ex.rename_axis(index={'Upper':'Changed_row'}, columns={'Other':'Changed_Col'}).head()
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
      <th></th>
      <th>Big</th>
      <th colspan="4" halign="left">C</th>
      <th colspan="4" halign="left">D</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Small</th>
      <th colspan="2" halign="left">c</th>
      <th colspan="2" halign="left">d</th>
      <th colspan="2" halign="left">c</th>
      <th colspan="2" halign="left">d</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Changed_Col</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
    </tr>
    <tr>
      <th>Changed_row</th>
      <th>Lower</th>
      <th>Extra</th>
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
      <th rowspan="4" valign="top">A</th>
      <th rowspan="2" valign="top">a</th>
      <th>alpha</th>
      <td>3</td>
      <td>6</td>
      <td>-9</td>
      <td>-6</td>
      <td>-6</td>
      <td>-2</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-5</td>
      <td>-3</td>
      <td>3</td>
      <td>-8</td>
      <td>-3</td>
      <td>-2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>alpha</th>
      <td>-4</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>7</td>
      <td>-4</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-9</td>
      <td>9</td>
      <td>-6</td>
      <td>8</td>
      <td>5</td>
      <td>-2</td>
      <td>-9</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>B</th>
      <th>a</th>
      <th>alpha</th>
      <td>0</td>
      <td>-9</td>
      <td>1</td>
      <td>-6</td>
      <td>2</td>
      <td>9</td>
      <td>-7</td>
      <td>-9</td>
    </tr>
  </tbody>
</table>
</div>



使用函数实现：


```python
df_ex.rename_axis(index=lambda x: 'Changed_row' if x == 'Upper' else x, 
                  columns=lambda x: 'Changed_Col' if x == 'Other' else x).head()
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
      <th></th>
      <th>Big</th>
      <th colspan="4" halign="left">C</th>
      <th colspan="4" halign="left">D</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Small</th>
      <th colspan="2" halign="left">c</th>
      <th colspan="2" halign="left">d</th>
      <th colspan="2" halign="left">c</th>
      <th colspan="2" halign="left">d</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Changed_Col</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
      <th>cat</th>
      <th>dog</th>
    </tr>
    <tr>
      <th>Changed_row</th>
      <th>Lower</th>
      <th>Extra</th>
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
      <th rowspan="4" valign="top">A</th>
      <th rowspan="2" valign="top">a</th>
      <th>alpha</th>
      <td>3</td>
      <td>6</td>
      <td>-9</td>
      <td>-6</td>
      <td>-6</td>
      <td>-2</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-5</td>
      <td>-3</td>
      <td>3</td>
      <td>-8</td>
      <td>-3</td>
      <td>-2</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>alpha</th>
      <td>-4</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>7</td>
      <td>-4</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-9</td>
      <td>9</td>
      <td>-6</td>
      <td>8</td>
      <td>5</td>
      <td>-2</td>
      <td>-9</td>
      <td>-8</td>
    </tr>
    <tr>
      <th>B</th>
      <th>a</th>
      <th>alpha</th>
      <td>0</td>
      <td>-9</td>
      <td>1</td>
      <td>-6</td>
      <td>2</td>
      <td>9</td>
      <td>-7</td>
      <td>-9</td>
    </tr>
  </tbody>
</table>
</div>



## 3 练习
### 3.1 Ex1：公司员工数据集
现有一份公司员工数据集：


```python
df = pd.read_csv('../data/company.csv')
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
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
      <th>department</th>
      <th>job_title</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>CEO</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>VP Stores</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>Legal Counsel</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>



1. 分别只使用`query`和`loc`选出年龄不超过四十岁且工作部门为`Dairy`或`Bakery`的男性。
2. 选出员工`ID`号 为奇数所在行的第1、第3和倒数第2列。
3. 按照以下步骤进行索引操作：

* 把后三列设为索引后交换内外两层
* 恢复中间一层
* 修改外层索引名为`Gender`
* 用下划线合并两层行索引
* 把行索引拆分为原状态
* 修改索引名为原表名称
* 恢复默认索引并将列保持为原表的相对位置

**我的解答：**

**第1问：**


```python
# 使用loc选择器
df.loc[(df.age < 40) & (df.department.isin(['Dairy', 'Bakery'])) & (df.gender == 'M')].head()
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
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
      <th>department</th>
      <th>job_title</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3722</th>
      <td>5902</td>
      <td>1/12/1976</td>
      <td>39</td>
      <td>New Westminster</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3724</th>
      <td>5904</td>
      <td>1/16/1976</td>
      <td>39</td>
      <td>Kelowna</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3725</th>
      <td>5905</td>
      <td>1/19/1976</td>
      <td>39</td>
      <td>Burnaby</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3727</th>
      <td>5907</td>
      <td>1/30/1976</td>
      <td>39</td>
      <td>Cranbrook</td>
      <td>Bakery</td>
      <td>Baker</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3730</th>
      <td>5910</td>
      <td>2/5/1976</td>
      <td>39</td>
      <td>New Westminster</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 使用query方法
df.query('age < 40 & department == ["Dairy", "Bakery"] & gender == "M"').head()
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
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
      <th>department</th>
      <th>job_title</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3722</th>
      <td>5902</td>
      <td>1/12/1976</td>
      <td>39</td>
      <td>New Westminster</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3724</th>
      <td>5904</td>
      <td>1/16/1976</td>
      <td>39</td>
      <td>Kelowna</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3725</th>
      <td>5905</td>
      <td>1/19/1976</td>
      <td>39</td>
      <td>Burnaby</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3727</th>
      <td>5907</td>
      <td>1/30/1976</td>
      <td>39</td>
      <td>Cranbrook</td>
      <td>Bakery</td>
      <td>Baker</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3730</th>
      <td>5910</td>
      <td>2/5/1976</td>
      <td>39</td>
      <td>New Westminster</td>
      <td>Dairy</td>
      <td>Dairy Person</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>



**第2问：**  
根据题意，采用`iloc`索引器，根据过滤条件`df.EmployeeID%2==1`，选取`[0, 2, -2]`列


```python
df.iloc[(df.EmployeeID % 2 == 1).values, [0, 2, -2]].head()
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
      <th>EmployeeID</th>
      <th>age</th>
      <th>job_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1319</td>
      <td>58</td>
      <td>VP Stores</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1321</td>
      <td>56</td>
      <td>VP Human Resources</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1323</td>
      <td>53</td>
      <td>Exec Assistant, VP Stores</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1325</td>
      <td>51</td>
      <td>Exec Assistant, Legal Counsel</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1329</td>
      <td>48</td>
      <td>Store Manager</td>
    </tr>
  </tbody>
</table>
</div>



**第3问：**


```python
df_copy = df.copy()
```


```python
df_copy.columns[-3:].tolist()
```




    ['department', 'job_title', 'gender']




```python
# 把后三列设为索引后交换内外两层
df_copy = df_copy.set_index(df_copy.columns[-3:].tolist()).swaplevel(0,2)
df_copy.head()
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
      <th></th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>job_title</th>
      <th>department</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <th>CEO</th>
      <th>Executive</th>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">F</th>
      <th>VP Stores</th>
      <th>Executive</th>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Legal Counsel</th>
      <th>Executive</th>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">M</th>
      <th>VP Human Resources</th>
      <th>Executive</th>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>VP Finance</th>
      <th>Executive</th>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 恢复中间一层
df_copy = df_copy.reset_index(level=1)
df_copy.head()
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
      <th>job_title</th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>department</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <th>Executive</th>
      <td>CEO</td>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">F</th>
      <th>Executive</th>
      <td>VP Stores</td>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>Legal Counsel</td>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">M</th>
      <th>Executive</th>
      <td>VP Human Resources</td>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>VP Finance</td>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改外层索引名为Gender
df_copy = df_copy.rename_axis(index={'gender':'Gender'})
df_copy.head()
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
      <th>job_title</th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>department</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <th>Executive</th>
      <td>CEO</td>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">F</th>
      <th>Executive</th>
      <td>VP Stores</td>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>Legal Counsel</td>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">M</th>
      <th>Executive</th>
      <td>VP Human Resources</td>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>VP Finance</td>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用下划线合并两层行索引
df_copy.index = df_copy.index.map(lambda x: '_'.join(x))
df_copy.head()
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
      <th>job_title</th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M_Executive</th>
      <td>CEO</td>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>F_Executive</th>
      <td>VP Stores</td>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>F_Executive</th>
      <td>Legal Counsel</td>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>M_Executive</th>
      <td>VP Human Resources</td>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>M_Executive</th>
      <td>VP Finance</td>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 把行索引拆分为原状态
df_copy.index = df_copy.index.map(lambda x:tuple(x.split('_')))
df_copy.head()
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
      <th>job_title</th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <th>Executive</th>
      <td>CEO</td>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">F</th>
      <th>Executive</th>
      <td>VP Stores</td>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>Legal Counsel</td>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">M</th>
      <th>Executive</th>
      <td>VP Human Resources</td>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>VP Finance</td>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改索引名为原表名称
df_copy = df_copy.rename_axis(['gender', 'department'], axis=0)
df_copy.head()
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
      <th>job_title</th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
    <tr>
      <th>gender</th>
      <th>department</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <th>Executive</th>
      <td>CEO</td>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">F</th>
      <th>Executive</th>
      <td>VP Stores</td>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>Legal Counsel</td>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">M</th>
      <th>Executive</th>
      <td>VP Human Resources</td>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>VP Finance</td>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 恢复默认索引并将列保持为原表的相对位置
df_copy = df_copy.reset_index()
df_copy.head()
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
      <th>gender</th>
      <th>department</th>
      <th>job_title</th>
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>Executive</td>
      <td>CEO</td>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>Executive</td>
      <td>VP Stores</td>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>Executive</td>
      <td>Legal Counsel</td>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>Executive</td>
      <td>VP Human Resources</td>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>Executive</td>
      <td>VP Finance</td>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
    </tr>
  </tbody>
</table>
</div>



发现顺序不对，于是采用reindex重置索引，将列名作为`columns`参数


```python
df_copy = df_copy.reindex(columns=df.columns)
df_copy.head()
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
      <th>EmployeeID</th>
      <th>birthdate_key</th>
      <th>age</th>
      <th>city_name</th>
      <th>department</th>
      <th>job_title</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1318</td>
      <td>1/3/1954</td>
      <td>61</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>CEO</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1319</td>
      <td>1/3/1957</td>
      <td>58</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>VP Stores</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1320</td>
      <td>1/2/1955</td>
      <td>60</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>Legal Counsel</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1321</td>
      <td>1/2/1959</td>
      <td>56</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>VP Human Resources</td>
      <td>M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1322</td>
      <td>1/9/1958</td>
      <td>57</td>
      <td>Vancouver</td>
      <td>Executive</td>
      <td>VP Finance</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
assert df_copy.equals(df)
```

### 3.2 Ex2：巧克力数据集
现有一份关于巧克力评价的数据集：


```python
df = pd.read_csv('../data/chocolate.csv')
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
      <th>Company</th>
      <th>Review\r\nDate</th>
      <th>Cocoa\r\nPercent</th>
      <th>Company\r\nLocation</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A. Morin</td>
      <td>2016</td>
      <td>63%</td>
      <td>France</td>
      <td>3.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A. Morin</td>
      <td>2015</td>
      <td>70%</td>
      <td>France</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A. Morin</td>
      <td>2015</td>
      <td>70%</td>
      <td>France</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>
</div>



1. 把列索引名中的`\n`替换为空格。
2. 巧克力`Rating`评分为1至5，每0.25分一档，请选出2.75分及以下且可可含量`Cocoa Percent`高于中位数的样本。
3. 将`Review Date`和`Company Location`设为索引后，选出`Review Date`在2012年之后且`Company Location`不属于`France, Canada, Amsterdam, Belgium`的样本。

**我的解答：**  
**第1问:**


```python
df_demo = df.rename(columns=lambda x:str.replace(x, '\r\n', ' '))
df_demo.head()
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
      <th>Company</th>
      <th>Review Date</th>
      <th>Cocoa Percent</th>
      <th>Company Location</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A. Morin</td>
      <td>2016</td>
      <td>63%</td>
      <td>France</td>
      <td>3.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A. Morin</td>
      <td>2015</td>
      <td>70%</td>
      <td>France</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A. Morin</td>
      <td>2015</td>
      <td>70%</td>
      <td>France</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A. Morin</td>
      <td>2015</td>
      <td>70%</td>
      <td>France</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A. Morin</td>
      <td>2015</td>
      <td>70%</td>
      <td>France</td>
      <td>3.50</td>
    </tr>
  </tbody>
</table>
</div>



**第2问：**


```python
df_demo['Cocoa Percent'] = df_demo['Cocoa Percent'].apply(lambda x: float(x[:-1])/100)
```


```python
df_demo.query('Rating <=2.75 & `Cocoa Percent` > `Cocoa Percent`.median()').head()
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
      <th>Company</th>
      <th>Review Date</th>
      <th>Cocoa Percent</th>
      <th>Company Location</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Akesson's (Pralus)</td>
      <td>2010</td>
      <td>0.75</td>
      <td>Switzerland</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Akesson's (Pralus)</td>
      <td>2010</td>
      <td>0.75</td>
      <td>Switzerland</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Alain Ducasse</td>
      <td>2014</td>
      <td>0.75</td>
      <td>France</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Alain Ducasse</td>
      <td>2013</td>
      <td>0.75</td>
      <td>France</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Alain Ducasse</td>
      <td>2013</td>
      <td>0.75</td>
      <td>France</td>
      <td>2.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_demo[(df_demo['Rating'] <=2.75) & (df_demo['Cocoa Percent'] > df_demo['Cocoa Percent'].median())].head()
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
      <th>Company</th>
      <th>Review Date</th>
      <th>Cocoa Percent</th>
      <th>Company Location</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>Akesson's (Pralus)</td>
      <td>2010</td>
      <td>0.75</td>
      <td>Switzerland</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Akesson's (Pralus)</td>
      <td>2010</td>
      <td>0.75</td>
      <td>Switzerland</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Alain Ducasse</td>
      <td>2014</td>
      <td>0.75</td>
      <td>France</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Alain Ducasse</td>
      <td>2013</td>
      <td>0.75</td>
      <td>France</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Alain Ducasse</td>
      <td>2013</td>
      <td>0.75</td>
      <td>France</td>
      <td>2.50</td>
    </tr>
  </tbody>
</table>
</div>



**第3问：**


```python
idx = pd.IndexSlice
```


```python
# 设置Review Date和Company Location为索引
df_demo = df_demo.set_index(['Review Date', 'Company Location']).sort_index(level=0)
df_demo.head()
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
      <th>Company</th>
      <th>Cocoa Percent</th>
      <th>Rating</th>
    </tr>
    <tr>
      <th>Review Date</th>
      <th>Company Location</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2006</th>
      <th>Belgium</th>
      <td>Cote d' Or (Kraft)</td>
      <td>0.70</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>Dolfin (Belcolade)</td>
      <td>0.70</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>Neuhaus (Callebaut)</td>
      <td>0.73</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>Neuhaus (Callebaut)</td>
      <td>0.75</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>Neuhaus (Callebaut)</td>
      <td>0.71</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选出Review Date在2012年之后且Company Location不属于France, Canada, Amsterdam, Belgium的样本
df_demo.loc[idx[2012:, df_demo.index.get_level_values(1).difference(['France', 'Canada', 'Amsterdam', 'Belgium'])], :].head()
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
      <th>Company</th>
      <th>Cocoa Percent</th>
      <th>Rating</th>
    </tr>
    <tr>
      <th>Review Date</th>
      <th>Company Location</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2012</th>
      <th>Australia</th>
      <td>Bahen &amp; Co.</td>
      <td>0.70</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>Bahen &amp; Co.</td>
      <td>0.70</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>Bahen &amp; Co.</td>
      <td>0.70</td>
      <td>2.50</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>Cravve</td>
      <td>0.75</td>
      <td>3.25</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>Cravve</td>
      <td>0.65</td>
      <td>3.25</td>
    </tr>
  </tbody>
</table>
</div>


