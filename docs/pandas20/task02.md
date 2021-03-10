# Task02 Pandas基础 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 文件的读取与写入

通过使用`parse_datas`参数，将日期进行格式化


```python
import pandas as pd

pd.read_csv('../data/my_csv.csv', parse_dates=['col5'])
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020-01-07</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 常用基本函数

#### 1.2.1 `info`和`describe`函数


```python
df = pd.read_csv('../data/learn_pandas.csv')
df.columns
```




    Index(['School', 'Grade', 'Name', 'Gender', 'Height', 'Weight', 'Transfer',
           'Test_Number', 'Test_Date', 'Time_Record'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   School       200 non-null    object 
     1   Grade        200 non-null    object 
     2   Name         200 non-null    object 
     3   Gender       200 non-null    object 
     4   Height       183 non-null    float64
     5   Weight       189 non-null    float64
     6   Transfer     188 non-null    object 
     7   Test_Number  200 non-null    int64  
     8   Test_Date    200 non-null    object 
     9   Time_Record  200 non-null    object 
    dtypes: float64(2), int64(1), object(7)
    memory usage: 15.8+ KB
    

主要展示数据集里面的列名、非空个数、对应的数据类型，统计列类型的个数和数据集占用的内存


```python
df.describe()
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
      <th>Height</th>
      <th>Weight</th>
      <th>Test_Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>183.000000</td>
      <td>189.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>163.218033</td>
      <td>55.015873</td>
      <td>1.645000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.608879</td>
      <td>12.824294</td>
      <td>0.722207</td>
    </tr>
    <tr>
      <th>min</th>
      <td>145.400000</td>
      <td>34.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>157.150000</td>
      <td>46.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>161.900000</td>
      <td>51.000000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>167.500000</td>
      <td>65.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>193.900000</td>
      <td>89.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



&emsp;&emsp;主要展示数据集中类型为`float`和`int`的统计值，包括个数、均值、标准差、最小值、25%分位数、50%分位数（中位数）、75%分位数和最大值

#### 1.2.2 `drop_duplicates`函数
&emsp;&emsp;`drop_duplicates`用途是对重复数据进行数据清理。   
&emsp;&emsp;如果想要观察多个列组合的唯一值，可以使用`drop_duplicates`。其中的关键参数是`keep`，默认值`first`表示每个组合保留第一次出现的所在行，`last`表示保留最后一次出现的所在行，`False`表示把所有重复组合所在的行剔除。


```python
df_demo = df[['Gender','Transfer','Name']]
df_demo.drop_duplicates(['Gender', 'Transfer'])
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
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>N</td>
      <td>Gaopeng Yang</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>N</td>
      <td>Changqiang You</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Female</td>
      <td>NaN</td>
      <td>Peng You</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Male</td>
      <td>NaN</td>
      <td>Xiaopeng Shen</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Male</td>
      <td>Y</td>
      <td>Xiaojuan Qin</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Female</td>
      <td>Y</td>
      <td>Gaoli Feng</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.2.3 `replace`函数
`replace`函数可以进行方向替换，指定`method`参数为`ffill`则为用前面一个最近的未被替换的值进行替换，`bfill`则使用后面最近的未被替换的值进行替换。


```python
s = pd.Series(['a', 1, 'b', 2, 1, 1, 'a'])
s.replace([1, 2], method='ffill')
```




    0    a
    1    a
    2    b
    3    b
    4    b
    5    b
    6    a
    dtype: object



#### 1.2.4 `where`和`mask`函数  
`where`函数是根据条件进行反向过滤，`mask`函数是根据条件进行正向过滤。


```python
s = pd.Series([-1, 1.2345, 100, -50])
s.where(s<0, 100)
```




    0     -1.0
    1    100.0
    2    100.0
    3    -50.0
    dtype: float64




```python
s.mask(s<0, -100)
```




    0   -100.0000
    1      1.2345
    2    100.0000
    3   -100.0000
    dtype: float64



#### 1.2.5 窗口对象
1. 滑窗对象：`rolling`函数，可使用`window`参数设置滑动窗口


```python
s = pd.Series([1,2,3,4,5])
roller = s.rolling(window = 3)
roller.mean()
```




    0    NaN
    1    NaN
    2    2.0
    3    3.0
    4    4.0
    dtype: float64



2. 扩张窗口：又称为累计窗口，`expanding`函数


```python
s.expanding().sum()
```




    0     1.0
    1     3.0
    2     6.0
    3    10.0
    4    15.0
    dtype: float64



## 2 练一练

### 2.1 第1题
在`clip`中，超过边界的只能截断为边界值，如果要把超出边界的替换为自定义的值，应当如何做？


```python
import pandas as pd
s = pd.Series([-1, 1.2345, 100, -50])
```

**我的解答：**  
根据题意理解，假设要替换的值为(-2, 30)，其中下边界为-2、上边界为30，故得到的序列应该为`[-2, 1.2345, 30, -2]`


```python
def replace_clip(s, lower, upper, define_lower_value, define_upper_value):
    return s.mask(s<lower, define_lower_value).mask(s>upper, define_upper_value)
```


```python
replace_clip(s, lower=0, upper=2, define_lower_value=-2, define_upper_value=30)
```




    0    -2.0000
    1     1.2345
    2    30.0000
    3    -2.0000
    dtype: float64



### 2.2 第2题
`rolling`对象的默认窗口方向都是向前的，某些情况下用户需要向后的窗口，例如对1,2,3设定向后窗口为2的`sum`操作，结果为3,5,NaN，此时应该如何实现向后的滑窗操作？（提示：使用`shift`）

**我的解答：**  
例如对`[1,2,3,4,5]`设定向后窗口为3的`sum`操作，结果应为`[6,9,12,NaN,NaN]`


```python
s = pd.Series([1, 2, 3, 4, 5])
s.rolling(3).sum().shift(-2)
```




    0     6.0
    1     9.0
    2    12.0
    3     NaN
    4     NaN
    dtype: float64



### 2.3 第3题
`cummax`, `cumsum`, `cumprod`函数是典型的类扩张窗口函数，请使用`expanding`对象依次实现它们。

**我的解答：**


```python
s = pd.Series([5,4,3,2,1])
```

1. **累计求最大值**

函数`cummax`的执行结果：


```python
s.cummax()
```




    0    5
    1    5
    2    5
    3    5
    4    5
    dtype: int64



使用`expanding`实现的`cummax`：


```python
s.expanding().max()
```




    0    5.0
    1    5.0
    2    5.0
    3    5.0
    4    5.0
    dtype: float64



2. **累计求和**

函数`cumsum`的执行结果：


```python
s.cumsum()
```




    0     5
    1     9
    2    12
    3    14
    4    15
    dtype: int64



使用`expanding`实现的`cumsum`：


```python
s.expanding().sum()
```




    0     5.0
    1     9.0
    2    12.0
    3    14.0
    4    15.0
    dtype: float64



3. **累计求积**

函数`cumprod`的执行结果：


```python
s.cumprod()
```




    0      5
    1     20
    2     60
    3    120
    4    120
    dtype: int64



使用`expanding`实现的`cumprod`：


```python
import numpy as np
s.expanding().apply(lambda x: np.array(x).prod())
```




    0      5.0
    1     20.0
    2     60.0
    3    120.0
    4    120.0
    dtype: float64



## 3 练习

### 3.1 Ex1：口袋妖怪数据集
现有一份口袋妖怪的数据集，下面进行一些背景说明：

* `#`代表全国图鉴编号，不同行存在相同数字则表示为该妖怪的不同状态

* 妖怪具有单属性和双属性两种，对于单属性的妖怪，`Type 2`为缺失值
* `Total, HP, Attack, Defense, Sp.Atk, Sp.Def, Speed`分别代表种族值、体力、物攻、防御、特攻、特防、速度，其中种族值为后6项之和


```python
df = pd.read_csv('../data/pokemon.csv')
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
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



1. 对`HP, Attack, Defense, Sp. Atk, Sp. Def, Speed`进行加总，验证是否为`Total`值。

2. 对于`#`重复的妖怪只保留第一条记录，解决以下问题：

* 求第一属性的种类数量和前三多数量对应的种类
* 求第一属性和第二属性的组合种类
* 求尚未出现过的属性组合

3. 按照下述要求，构造`Series`：

* 取出物攻，超过120的替换为`high`，不足50的替换为`low`，否则设为`mid`
* 取出第一属性，分别用`replace`和`apply`替换所有字母为大写
* 求每个妖怪六项能力的离差，即所有能力中偏离中位数最大的值，添加到`df`并从大到小排序

**我的解答：**  
**第1问：**


```python
df = pd.read_csv('../data/pokemon.csv')
df[df['Total'] != df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].sum(1)]
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
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



通过观察，可知所有数据都是的6个属性的加总与`Total`值是一致的。

**第2问：**


```python
# 对'#'重复的妖怪只保留第一条记录
df_dropdup = df.drop_duplicates('#', keep='first')
df_dropdup.shape
```




    (721, 11)




```python
# 第一属性的种类数量
df_dropdup['Type 1'].nunique()
```




    18




```python
# 前三多数量对应的种类
df_dropdup['Type 1'].value_counts().head(3)
```




    Water     105
    Normal     93
    Grass      66
    Name: Type 1, dtype: int64




```python
# 第一属性和第二属性的组合种类
df_type1_type2 = df_dropdup.drop_duplicates(['Type 1', 'Type 2'])
df_type1_type2[['Type 1', 'Type 2']]
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
      <th>Type 1</th>
      <th>Type 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Grass</td>
      <td>Poison</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fire</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fire</td>
      <td>Flying</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Water</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bug</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>773</th>
      <td>Rock</td>
      <td>Fairy</td>
    </tr>
    <tr>
      <th>778</th>
      <td>Ghost</td>
      <td>Grass</td>
    </tr>
    <tr>
      <th>790</th>
      <td>Flying</td>
      <td>Dragon</td>
    </tr>
    <tr>
      <th>797</th>
      <td>Psychic</td>
      <td>Ghost</td>
    </tr>
    <tr>
      <th>799</th>
      <td>Fire</td>
      <td>Water</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 2 columns</p>
</div>




```python
# 求尚未出现过的属性组合
type1_list = df_dropdup['Type 1'].unique()
```


```python
type2_list = df_dropdup['Type 2'].unique()
# 删除nan的数据
type2_list = type2_list[~pd.isnull(type2_list)]
```


```python
import numpy as np
set_full = set([(i,j) if i!=j else None for i in type1_list for j in type2_list])
len(set_full)
```




    307




```python
set_used = set((i, j) if type(j) == str else None for i,j in zip(df_type1_type2['Type 1'], df_type1_type2['Type 2']))
len(set_used)
```




    126




```python
res = set_full.difference(set_used)
len(res)
```




    181



**第3问：**


```python
# 取出物攻，超过120的替换为high，不足50的替换为low，否则设为mid
df_attack = df['Attack'].copy()

def cond_replace(x):
     if x < 50:
        return 'low'
     elif x > 120:
        return 'high'
     return 'mid'
res = df_attack.apply(cond_replace)
res.head()
```




    0    low
    1    mid
    2    mid
    3    mid
    4    mid
    Name: Attack, dtype: object




```python
# 取出第一属性，分别用replace和apply替换所有字母为大写
# 使用replace实现
df_type1 = df['Type 1'].copy()
df_type1.replace(df_type1.unique(), [str.upper(s) for s in df_type1.unique()]).head()
```




    0    GRASS
    1    GRASS
    2    GRASS
    3    GRASS
    4     FIRE
    Name: Type 1, dtype: object




```python
# 使用apply实现
df_type1.apply(lambda x: str.upper(x)).head()
```




    0    GRASS
    1    GRASS
    2    GRASS
    3    GRASS
    4     FIRE
    Name: Type 1, dtype: object




```python
# 求每个妖怪六项能力的离差，即所有能力中偏离中位数最大的值，添加到df并从大到小排序
df['Capacity'] = df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].apply(lambda x: np.max((x-x.mean()).abs()), 1)
df.sort_values('Capacity',ascending=False)
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
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>Capacity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>121</th>
      <td>113</td>
      <td>Chansey</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>450</td>
      <td>250</td>
      <td>5</td>
      <td>5</td>
      <td>35</td>
      <td>105</td>
      <td>50</td>
      <td>175.000000</td>
    </tr>
    <tr>
      <th>261</th>
      <td>242</td>
      <td>Blissey</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>540</td>
      <td>255</td>
      <td>10</td>
      <td>10</td>
      <td>75</td>
      <td>135</td>
      <td>55</td>
      <td>165.000000</td>
    </tr>
    <tr>
      <th>230</th>
      <td>213</td>
      <td>Shuckle</td>
      <td>Bug</td>
      <td>Rock</td>
      <td>505</td>
      <td>20</td>
      <td>10</td>
      <td>230</td>
      <td>10</td>
      <td>230</td>
      <td>5</td>
      <td>145.833333</td>
    </tr>
    <tr>
      <th>224</th>
      <td>208</td>
      <td>SteelixMega Steelix</td>
      <td>Steel</td>
      <td>Ground</td>
      <td>610</td>
      <td>75</td>
      <td>125</td>
      <td>230</td>
      <td>55</td>
      <td>95</td>
      <td>30</td>
      <td>128.333333</td>
    </tr>
    <tr>
      <th>333</th>
      <td>306</td>
      <td>AggronMega Aggron</td>
      <td>Steel</td>
      <td>NaN</td>
      <td>630</td>
      <td>70</td>
      <td>140</td>
      <td>230</td>
      <td>60</td>
      <td>80</td>
      <td>50</td>
      <td>125.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>255</th>
      <td>236</td>
      <td>Tyrogue</td>
      <td>Fighting</td>
      <td>NaN</td>
      <td>210</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>383</th>
      <td>351</td>
      <td>Castform</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>420</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>358</th>
      <td>327</td>
      <td>Spinda</td>
      <td>Normal</td>
      <td>NaN</td>
      <td>360</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>550</th>
      <td>492</td>
      <td>ShayminLand Forme</td>
      <td>Grass</td>
      <td>NaN</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>165</th>
      <td>151</td>
      <td>Mew</td>
      <td>Psychic</td>
      <td>NaN</td>
      <td>600</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 12 columns</p>
</div>



### 3.2 Ex2：指数加权窗口
1. 作为扩张窗口的`ewm`窗口

在扩张窗口中，用户可以使用各类函数进行历史的累计指标统计，但这些内置的统计函数往往把窗口中的所有元素赋予了同样的权重。事实上，可以给出不同的权重来赋给窗口中的元素，指数加权窗口就是这样一种特殊的扩张窗口。

其中，最重要的参数是`alpha`，它决定了默认情况下的窗口权重为$w_i=(1−\alpha)^i,i\in\{0,1,...,t\}$，其中$i=t$表示当前元素，$i=0$表示序列的第一个元素。

从权重公式可以看出，离开当前值越远则权重越小，若记原序列为$x$，更新后的当前元素为$y_t$，此时通过加权公式归一化后可知：

$$
\begin{aligned}
y_t &=\frac{\sum_{i=0}^{t} w_i x_{t-i}}{\sum_{i=0}^{t} w_i} \\
&=\frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^{t} x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^{t-1}}
\end{aligned}
$$

对于`Series`而言，可以用`ewm`对象如下计算指数平滑后的序列：


```python
np.random.seed(0)
s = pd.Series(np.random.randint(-1,2,30).cumsum())
s.head()
```




    0   -1
    1   -1
    2   -2
    3   -2
    4   -2
    dtype: int32




```python
s.ewm(alpha=0.2).mean().head()
```




    0   -1.000000
    1   -1.000000
    2   -1.409836
    3   -1.609756
    4   -1.725845
    dtype: float64



请用`expanding`窗口实现。

2. 作为滑动窗口的`ewm`窗口

从第1问中可以看到，`ewm`作为一种扩张窗口的特例，只能从序列的第一个元素开始加权。现在希望给定一个限制窗口`n`，只对包含自身最近的`n`个窗口进行滑动加权平滑。请根据滑窗函数，给出新的`wi`与`yt`的更新公式，并通过`rolling`窗口实现这一功能。

**我的解答：**  

**第1问：**  
看到$\displaystyle \sum_{i=0}^{t} w_i x_{t-i}$，可想到用`numpy`的乘法，需要构造$w$和$x$矩阵：  
易知$x$矩阵只需要逆转即可，而$w$矩阵只需使用`(1 - alpha) ** np.arange(t)`，其中$t$表示$x$矩阵的长度


```python
def my_ewm(x, alpha):
    # 将x逆转
    x = np.array(x)[::-1]
    # 构造w
    w = (1 - alpha) ** np.arange(len(x))
    return (w * x).sum() / w.sum()

alpha = 0.2
s.expanding().apply(lambda x: my_ewm(x, alpha)).head()
```




    0   -1.000000
    1   -1.000000
    2   -1.409836
    3   -1.609756
    4   -1.725845
    dtype: float64



**第2问：**  
根据题意，限制窗口为`n`  
$w_i$的更新公式：$w_i=(1−\alpha)^i,i\in\{0,1,...,n-1\}$  
$y_t$的更新公式：
$$
\begin{aligned}
y_t &=\frac{\sum_{i=0}^{n-1} w_i x_{t-i}}{\sum_{i=0}^{n-1} w_i} \\
&=\frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^{n-1} x_{t-(n-1)}}
{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^{n-1}} 
\end{aligned}
$$


```python
def my_ewm(x, alpha):
    x = np.array(x)[::-1]
    w = (1 - alpha) ** np.arange(len(x))
    return (w * x).sum() / w.sum()

alpha = 0.2
s.rolling(window=3).apply(lambda x: my_ewm(x, alpha)).head()
```




    0         NaN
    1         NaN
    2   -1.409836
    3   -1.737705
    4   -2.000000
    dtype: float64


