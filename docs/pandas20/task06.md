# Task6 连接 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 关系型连接
- 连接方式主要有左连接`left`、右连接`right`、内连接`inner`、外连接`outer`
- 值连接通过`merge`函数实现，`on`为连接的键，`how`为连接方式
- 索引连接通过`join`函数实现，`on`为索引名，`how`为连接方式，`lsuffix`和`rsuffix`为左右后缀

### 1.2 方向连接
- `concat`函数：`axis`为连接方向，`join`为连接方式，`keys`为源表表名
- 序列与表合并，通过`append`函数（对行的添加）和`assign`函数（对列的添加，并返回DataFrame副本）

### 1.3 类连接操作
- `compare`函数：主要用于两表/两序列比较
- `combine`函数：主要用于组合，可传入自定义的规则函数，函数有两个参数`s1`,`s2`，分别来自于两张表对应的列

## 2 练一练


```python
import numpy as np
import pandas as pd
```

### 2.1 第1题
上面以多列为键的例子中，错误写法显然是一种多对多连接，而正确写法是一对一连接，请修改原表，使得以多列为键的正确写法能够通过`validate='1:m'`的检验，但不能通过`validate='m:1'`的检验。

**我的解答：**


```python
df1 = pd.DataFrame({'Name':['San Zhang', 'San Zhang'],
                    'Age':[20, 21],
                    'Class':['one', 'two']})
df1
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
      <th>Age</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Zhang</td>
      <td>20</td>
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>San Zhang</td>
      <td>21</td>
      <td>two</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 将df2中Class列数据修改为['one', 'one']，即一班有两个叫张三的人，一个男生一个女生
df2 = pd.DataFrame({'Name':['San Zhang', 'San Zhang'],
                    'Gender':['F', 'M'],
                    'Class':['one', 'one']})
df2
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
      <th>Gender</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Zhang</td>
      <td>F</td>
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>San Zhang</td>
      <td>M</td>
      <td>one</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.merge(df2, on=['Name', 'Class'], how='left', validate="1:m")
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
      <th>Age</th>
      <th>Class</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>San Zhang</td>
      <td>20</td>
      <td>one</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>San Zhang</td>
      <td>20</td>
      <td>one</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>San Zhang</td>
      <td>21</td>
      <td>two</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
try:
    df1.merge(df2, on=['Name', 'Class'], how='left', validate="m:1")
except Exception as e:
    print(e)
```

    Merge keys are not unique in right dataset; not a many-to-one merge
    

### 2.2 第2题 
请在上述代码的基础上修改，保留`df2`中4个未被`df1`替换的相应位置原始值。

**我的解答：**


```python
df1 = pd.DataFrame({'A':[1,2], 'B':[3,4], 'C':[5,6]})
df2 = pd.DataFrame({'B':[5,6], 'C':[7,8], 'D':[9,10]}, index=[1,2])
```


```python
df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2
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
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>7</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>8</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
def choose_min(s1, s2):
    s2 = s2.reindex_like(s1)
    res = s1.where(s1<s2, s2)
    # isna表示是否为缺失值，返回布尔序列（删除该行即可）
    #res = res.mask(s1.isna())
    return res

df1.combine(df2, choose_min)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3 第3题
除了`combine`之外，`pandas`中还有一个`combine_first`方法，其功能是在对两张表组合时，若第二张表中的值在第一张表中对应索引位置的值不是缺失状态，那么就使用第一张表的值填充。下面给出一个例子，请用`combine`函数完成相同的功能。


```python
df1 = pd.DataFrame({'A':[1,2], 'B':[3,np.nan]})
df2 = pd.DataFrame({'A':[5,6], 'B':[7,8]}, index=[1,2])
df1.combine_first(df2)
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



**我的解答：**


```python
def choose_min(s1, s2):
    s2 = s2.reindex_like(s1)
    res = s1.mask(s1.isna(), s2)
    return res

df1.combine(df2, choose_min)
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3 练习

### 3.1 Ex1：美国疫情数据集

现有美国4月12日至11月16日的疫情报表，数据见`notebook/pandas20/data/us_report`文件夹路径，请将`New York`的`Confirmed, Deaths, Recovered, Active`合并为一张表，索引为按如下方法生成的日期字符串序列：


```python
date = pd.date_range('20200412', '20201116').to_series()
date = date.dt.month.astype('string').str.zfill(2) +'-'+ date.dt.day.astype('string').str.zfill(2) +'-'+ '2020'
date = date.tolist()
date[:5]
```




    ['04-12-2020', '04-13-2020', '04-14-2020', '04-15-2020', '04-16-2020']



**我的解答：**

1. 先进行尝试，读取两个文件，最后通过`concat`函数进行合并


```python
df1 = pd.read_csv('../data/us_report/04-12-2020.csv', usecols=['Confirmed', 'Deaths', 'Recovered', 'Active', 'Province_State'])
df2 = pd.read_csv('../data/us_report/04-13-2020.csv', usecols=['Confirmed', 'Deaths', 'Recovered', 'Active', 'Province_State'])
df1 = df1[df1['Province_State']=='New York']
df2 = df2[df2['Province_State']=='New York']
```


```python
pd.concat([df1, df2])
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
      <th>Province_State</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>New York</td>
      <td>189033</td>
      <td>9385</td>
      <td>23887.0</td>
      <td>179648.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>New York</td>
      <td>195749</td>
      <td>10058</td>
      <td>23887.0</td>
      <td>185691.0</td>
    </tr>
  </tbody>
</table>
</div>



2. 整合代码，并修改索引列


```python
concat_list = []

for d in date:
    df = pd.read_csv('../data/us_report/' + d + '.csv', usecols=['Confirmed', 'Deaths', 'Recovered', 'Active', 'Province_State'])
    df = df[df['Province_State']=='New York'].iloc[:,1:]
    concat_list.append(df)

res = pd.concat(concat_list)
res.index = date
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>04-12-2020</th>
      <td>189033</td>
      <td>9385</td>
      <td>23887.0</td>
      <td>179648.0</td>
    </tr>
    <tr>
      <th>04-13-2020</th>
      <td>195749</td>
      <td>10058</td>
      <td>23887.0</td>
      <td>185691.0</td>
    </tr>
    <tr>
      <th>04-14-2020</th>
      <td>203020</td>
      <td>10842</td>
      <td>23887.0</td>
      <td>192178.0</td>
    </tr>
    <tr>
      <th>04-15-2020</th>
      <td>214454</td>
      <td>11617</td>
      <td>23887.0</td>
      <td>202837.0</td>
    </tr>
    <tr>
      <th>04-16-2020</th>
      <td>223691</td>
      <td>14832</td>
      <td>23887.0</td>
      <td>208859.0</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 Ex2：实现join函数

请实现带有`how`参数的`join`函数

* 假设连接的两表无公共列
* 调用方式为 `join(df1, df2, how="left")`
* 给出测试样例

**我的解答：**


```python
def join(df1: pd.DataFrame, df2: pd.DataFrame, how='left') -> pd.DataFrame:
    # 得到所有要连接的列
    res_col = df1.columns.tolist() + df2.columns.tolist()
    # 得到无重复的索引
    index_dup = df1.index.unique().intersection(df2.index.unique())
    # 建立空的DataFrame用于连接
    res_df = pd.DataFrame(columns=res_col)

    # 构造笛卡尔积的DataFrame
    for label in index_dup:
         # 防止单个str字符串被list()函数拆成单个字符列表
        df1_values = df1.loc[label].values.reshape(-1, 1)
        df2_values = df2.loc[label].values.reshape(-1, 1)

        # 计算笛卡尔积
        cartesian = [list(i) + list(j) for i in df1_values for j in df2_values]
        # 构造笛卡尔积的DataFrame
        res_df = create_dateframe(cartesian, label, res_col, res_df)

    if how in ['left', 'outer']:
        # 遍历df1，进行连接
        for label in df1.index.unique().difference(index_dup):
            df_lable = validate_dataframe(df1, label)
            cat = [list(i) + [np.nan] * df2.shape[1] for i in df_lable.values]
            # 构建DataFrame
            res_df = create_dateframe(cat, label, res_col, res_df)
    if how in ['right', 'outer']:
        # 遍历df2，进行连接
        for label in df2.index.unique().difference(index_dup):
            df_lable = validate_dataframe(df2, label)
            cat = [[np.nan] * df1.shape[1] + list(i) for i in df_lable.values]
            # 构建DataFrame
            res_df = create_dateframe(cat, label, res_col, res_df)

    res_df = columns_dtype_convert(res_df, df1, df2)
    return res_df


def validate_dataframe(df: pd.DataFrame, label) -> pd.DataFrame:
    '''
    判断df.loc[lable]是否为DataFrame，如果不是将其转换为DataFrame
    :param df:
    :param label:
    :return:
    '''
    df_lable = df.loc[label]
    if not isinstance(df_lable, pd.DataFrame):
        df_lable = df.loc[label].to_frame()
    return df_lable


def create_dateframe(data, label, res_col, res_df: pd.DataFrame) -> pd.DataFrame:
    dup_df = pd.DataFrame(data, index=[label] * len(data), columns=res_col)
    res_df = pd.concat([res_df, dup_df])
    return res_df


def columns_dtype_convert(res_df: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # 进行类型转换
    df1_dtypes = dict(df1.dtypes)
    df2_dtypes = dict(df2.dtypes)
    df1_dtypes.update(df2_dtypes)

    res_number_cols = df1.select_dtypes('number').columns.tolist() + df2.select_dtypes('number').columns.tolist()
    # 只留下数据里面有nan的列
    nan_cols = []
    for col in res_number_cols:
        if res_df[col].isna().any():
            nan_cols.append(col)
    # 将number类型的数据转换为float64类型数据
    res_df = res_df.astype(dict(zip(nan_cols, ['float64'] * len(nan_cols))))
    # 在转换列中删除数据为nan的列
    if nan_cols:
        df1_dtypes.pop(*nan_cols)
    res_df = res_df.astype(df1_dtypes)

    return res_df
```


```python
def test(df1, df2, how):
    df = df1.join(df2, how=how)
    res = join(df1, df2, how=how)
    assert res.equals(df)


df1 = pd.DataFrame(columns=['Name'], data=['LiSi', 'ZhangSan', 'WangWu', 'XiaoLiu'],
                   index=['A', 'B', 'B', 'C'])
df2 = pd.DataFrame(columns=['Grade'], data=[92, 64, 73, 80],
                   index=['A', 'A', 'B', 'D'])

test(df1, df2, 'left')
test(df1, df2, 'right')
test(df1, df2, 'inner')
test(df1, df2, 'outer')
```
