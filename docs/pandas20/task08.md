# Task8 文本数据 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 str对象
- Series的str对象
- str[]索引器：取出某个位置的元素
- string类型：序列中至少有一个可迭代（Iterable）对象，包括但不限于字符串、字典、列表

### 1.2 正则表达式

|元字符 |   描述 |
| :-----| ----: |
|.       |    匹配除换行符以外的任意字符|
|\[ \]     |      字符类，匹配方括号中包含的任意字符|
|\[^ \]     |      否定字符类，匹配方括号中不包含的任意字符|
|\*       |    匹配前面的子表达式零次或多次|
|\+       |    匹配前面的子表达式一次或多次|
|?        |   匹配前面的子表达式零次或一次|
|{n,m}    |       花括号，匹配前面字符至少 n 次，但是不超过 m 次|
|(xyz)   |        字符组，按照确切的顺序匹配字符xyz|
|\|     |      分支结构，匹配符号之前的字符或后面的字符|
|\\    |       转义符，它可以还原元字符原来的含义|
|^    |       匹配行的开始|
|$   |        匹配行的结束|

|简写    |  描述 |
| :-----| :---- |
|\\w     |   匹配所有字母、数字、下划线: \[a-zA-Z0-9\_\] |
|\\W     |   匹配非字母和数字的字符: \[^\\w\]|
|\\d     |   匹配数字: \[0-9\]|
|\\D   |     匹配非数字: \[^\\d\]|
|\\s    |    匹配空格符: \[\\t\\n\\f\\r\\p{Z}\]|
|\\S    |    匹配非空格符: \[^\\s\]|
|\\B  |      匹配一组非空字符开头或结尾的位置，不代表具体字符|

### 1.3 文本处理的五类操作
- 拆分：`str.split`方法进行字符串的列拆分
- 合并：`str.join`和`str.cat`方法进行字符串列表连接
- 匹配：
| 方法名 | 是否支持<br>正则表达式 | 描述 |
| :--- | :--- | :--- |
|`str.contains` | 是 | 用于判断每个字符串是否包含正则表达式的字符 |
|`str.startswith` | 否 | 用于判断是否以字符串为开始 |
|`str.endswith` | 否 | 用于判断是否以字符串为结束 |
|`str.match`| 是 | 每个字符串起始处是否符合给定正则表达式的字符 |
|`str.find`| 否 | 从左到右第一次匹配的位置的索引 |
|`str.rfind`| 否 | 从右到左第一次匹配的位置的索引 |

- 替换：`str.replace`，可使用自定义的替换函数来处理
- 提取：`str.extract`只匹配一次，可返回DataFrame；`str.extractall`匹配所有符合条件的字符串

### 1.4 常用字符串函数
- 字母型函数：`upper`, `lower`, `title`, `capitalize`, `swapcase`
- 数值型函数：`pd.to_numeric`用于对字符格式的数值进行快速转换和筛选
- 统计型函数：`count`出现正则模式的次数，`len`字符串的长度
- 格式型函数：`strip`, `rstrip`, `lstrip`进行去除空格  
    `pad`, `rjust`, `ljust`, `center`进行填充

## 2 练习

### 2.1 Ex1：房屋信息数据集
现有一份房屋信息数据集如下：


```python
import pandas as pd
import numpy as np

df = pd.read_excel('../data/house_info.xls', usecols=['floor','year','area','price'])
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
      <th>floor</th>
      <th>year</th>
      <th>area</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>高层（共6层）</td>
      <td>1986年建</td>
      <td>58.23㎡</td>
      <td>155万</td>
    </tr>
    <tr>
      <th>1</th>
      <td>中层（共20层）</td>
      <td>2020年建</td>
      <td>88㎡</td>
      <td>155万</td>
    </tr>
    <tr>
      <th>2</th>
      <td>低层（共28层）</td>
      <td>2010年建</td>
      <td>89.33㎡</td>
      <td>365万</td>
    </tr>
  </tbody>
</table>
</div>



1. 将`year`列改为整数年份存储。
2. 将`floor`列替换为`Level, Highest`两列，其中的元素分别为`string`类型的层类别（高层、中层、低层）与整数类型的最高层数。
3. 计算房屋每平米的均价`avg_price`，以`***元/平米`的格式存储到表中，其中`***`为整数。

**我的解答：**

**第1问：**


```python
# 利用pd.to_numeric函数对数值型数据进行快速转换
df.year = pd.to_numeric(df.year.str[:4], errors='ignore').astype('Int64')
```


```python
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
      <th>floor</th>
      <th>year</th>
      <th>area</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>高层（共6层）</td>
      <td>1986</td>
      <td>58.23㎡</td>
      <td>155万</td>
    </tr>
    <tr>
      <th>1</th>
      <td>中层（共20层）</td>
      <td>2020</td>
      <td>88㎡</td>
      <td>155万</td>
    </tr>
    <tr>
      <th>2</th>
      <td>低层（共28层）</td>
      <td>2010</td>
      <td>89.33㎡</td>
      <td>365万</td>
    </tr>
    <tr>
      <th>3</th>
      <td>低层（共20层）</td>
      <td>2014</td>
      <td>82㎡</td>
      <td>308万</td>
    </tr>
    <tr>
      <th>4</th>
      <td>高层（共1层）</td>
      <td>2015</td>
      <td>98㎡</td>
      <td>117万</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 检查year的数据类型为Int64
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 31568 entries, 0 to 31567
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   floor   31219 non-null  object
     1   year    12850 non-null  Int64 
     2   area    31568 non-null  object
     3   price   31568 non-null  object
    dtypes: Int64(1), object(3)
    memory usage: 1017.5+ KB
    

**第2问：**


```python
import re

# 先用正则表达式进行尝试
pat = '(?P<Level>\w层)（共(?P<Highest>\d+)层）'
re.findall(pat, df.floor[0])
```




    [('高层', '6')]




```python
# 使用str.extract()函数进行抽取
df.floor.str.extract(pat).head()
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
      <th>Level</th>
      <th>Highest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>高层</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>中层</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>低层</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>低层</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>高层</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 进行数据列连接
df = pd.concat([df.drop(columns='floor'), df.floor.str.extract(pat)], axis=1)
```


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
      <th>year</th>
      <th>area</th>
      <th>price</th>
      <th>Level</th>
      <th>Highest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1986</td>
      <td>58.23㎡</td>
      <td>155万</td>
      <td>高层</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>88㎡</td>
      <td>155万</td>
      <td>中层</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>89.33㎡</td>
      <td>365万</td>
      <td>低层</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改数据类型
df.Level = df.Level.astype('string')
df.Highest = pd.to_numeric(df.Highest).astype('Int64')
```


```python
# 检查Level列和Highest列的数据类型
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 31568 entries, 0 to 31567
    Data columns (total 5 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   year     12850 non-null  Int64 
     1   area     31568 non-null  object
     2   price    31568 non-null  object
     3   Level    31219 non-null  string
     4   Highest  31219 non-null  Int64 
    dtypes: Int64(2), object(2), string(1)
    memory usage: 1.3+ MB
    

**第3问：**


```python
area_series = pd.to_numeric(df.area.str[:-1])
price_series = pd.to_numeric(df.price.str[:-1])
```


```python
df['avg_price'] = (round(price_series * 10000 / area_series, 2)).astype('string') + '元/平米'
```


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
      <th>year</th>
      <th>area</th>
      <th>price</th>
      <th>Level</th>
      <th>Highest</th>
      <th>avg_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1986</td>
      <td>58.23㎡</td>
      <td>155万</td>
      <td>高层</td>
      <td>6</td>
      <td>26618.58元/平米</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>88㎡</td>
      <td>155万</td>
      <td>中层</td>
      <td>20</td>
      <td>17613.64元/平米</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>89.33㎡</td>
      <td>365万</td>
      <td>低层</td>
      <td>28</td>
      <td>40859.73元/平米</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 Ex2：《权力的游戏》剧本数据集
现有一份权力的游戏剧本数据集如下：


```python
df = pd.read_csv('../data/script.csv')
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
      <th>Release Date</th>
      <th>Season</th>
      <th>Episode</th>
      <th>Episode Title</th>
      <th>Name</th>
      <th>Sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-04-17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>waymar royce</td>
      <td>What do you expect? They're savages. One lot s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-04-17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>will</td>
      <td>I've never seen wildlings do a thing like this...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-04-17</td>
      <td>Season 1</td>
      <td>Episode 1</td>
      <td>Winter is Coming</td>
      <td>waymar royce</td>
      <td>How close did you get?</td>
    </tr>
  </tbody>
</table>
</div>



1. 计算每一个`Episode`的台词条数。
2. 以空格为单词的分割符号，请求出单句台词平均单词量最多的前五个人。
3. 若某人的台词中含有问号，那么下一个说台词的人即为回答者。若上一人台词中含有n个问号，则认为回答者回答了n个问题，请求出回答最多问题的前五个人。

**我的解答：**

**第1问：**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23911 entries, 0 to 23910
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   Release Date   23911 non-null  object
     1    Season        23911 non-null  object
     2   Episode        23911 non-null  object
     3   Episode Title  23911 non-null  object
     4   Name           23908 non-null  object
     5   Sentence       23911 non-null  object
    dtypes: object(6)
    memory usage: 1.1+ MB
    


```python
# 可观察到Season有空格，故需要str.strip()去除两侧空格
df.columns = df.columns.str.strip()
```


```python
df.groupby(['Season', 'Episode'])['Sentence'].count()
```




    Season    Episode   
    Season 1  Episode 1     327
              Episode 10    266
              Episode 2     283
              Episode 3     353
              Episode 4     404
                           ... 
    Season 8  Episode 2     405
              Episode 3     155
              Episode 4      51
              Episode 5     308
              Episode 6     240
    Name: Sentence, Length: 73, dtype: int64



**第2问：**


```python
# 设置Name为行索引，得到平均单词量
res2_series = df.set_index('Name')['Sentence'].str.split(' ').str.len().groupby('Name').mean()
```


```python
# 通过排序得到最多的前5个人
res2_series.sort_values(ascending=False)[:5].index.tolist()
```




    ['male singer',
     'slave owner',
     'manderly',
     'lollys stokeworth',
     'dothraki matron']



**第3问：**


```python
# 得到后一个人的名字
res3_series = pd.Series(index=df.Name.shift(-1), data=df.Sentence.values)
res3_series
```




    Name
    will                What do you expect? They're savages. One lot s...
    waymar royce        I've never seen wildlings do a thing like this...
    will                                           How close did you get?
    gared                                         Close as any man would.
    royce                                We should head back to the wall.
                                              ...                        
    bronn               I think we can all agree that ships take prece...
    tyrion lannister        I think that's a very presumptuous statement.
    man                 I once brought a jackass and a honeycomb into ...
    all                                           The Queen in the North!
    NaN                 The Queen in the North! The Queen in the North...
    Length: 23911, dtype: object




```python
# 根据?进行统计，然后求和排序
res = res3_series.str.count('\?').groupby('Name').sum().sort_values(ascending=False)
res
```




    Name
    tyrion lannister    527
    jon snow            374
    jaime lannister     283
    arya stark          265
    cersei lannister    246
                       ... 
    male singer           0
    main                  0
    mace tyrell           0
    lyanna mormont        0
    young rodrik          0
    Length: 564, dtype: int64




```python
# 得到回答数最多的前5人
res[:5].index.tolist()
```




    ['tyrion lannister',
     'jon snow',
     'jaime lannister',
     'arya stark',
     'cersei lannister']


