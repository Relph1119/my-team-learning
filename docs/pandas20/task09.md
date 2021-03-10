# Task9 分类数据 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 `cat`对象
- `cat`对象的属性：`categories`（分类类别），`ordered`（是否有序），`codes`（整数编码）
- 类别的增加：`add_categories`函数
- 类别的删除：`remove_categories`函数，移除未出现的类别可使用`remove_unused_categories`函数
- 类别的设置：`set_categories`函数，原来的类别中如果存在元素不属于新类别，会被设置为缺失
- 类别的修改：`rename_categories`函数，传入字典

### 1.2 有序分类
- 无序->有序：`cat.reorder_categories`函数，参数`ordered`设置是否有序
- 有序->无序：`as_unordered`函数
- 排序：把列的类型修改为`category`后，再赋予相应的大小关系，可以使用`sort_index`和`sort_values`函数进行排序

### 1.3 区间类别
- `cut`函数：参数`bins`表示把整个数组按照数据中的最大值和最小值进行等间距的划分
- `qcut`函数：参数`q`表示把整个数组按照分位数进行划分

### 1.4 区间构造
- `pd.Interval`函数：可设置开闭状态
- `overlaps`函数：判断两个区间是否有交集
- `pd.interval_range`函数：构造一个等差区间，使用`start`, `end`, `periods`, `freq`四个参数中的三个即可

### 1.5 区间的属性和方法
- 可使用[]进行切片读取
- `left`属性：左端点
- `right`属性：右端点
- `mid`属性：两端点均值
- `lenght`属性：区间长度
- `contains`函数：逐个判断每个区间是否包含某元素
- `overlaps`函数：判断是否和一个pd.Interval对象有交集

## 2 练一练

### 2.1 第1题
无论是`interval_range`还是下一章时间序列中的`date_range`都是给定了等差序列中四要素中的三个，从而确定整个序列。请回顾等差数列中的首项、末项、项数和公差的联系，写出`interval_range`中四个参数之间的恒等关系。

**解答：**

&emsp;&emsp;可知等差数列的通项公式为$a_n = a_1 + (n - 1) \times d$，其中首项为$a_1$，末项为$a_n$，项数为$n$，公差为$d$  
&emsp;&emsp;`interval_range`函数中的`start`, `end`, `periods`, `freq`参数的恒等公式如下：

$$\text{end} = \text{start} + (\text{periods} - 1) \times \text{freq}$$

## 3 练习


```python
import pandas as pd
import numpy as np
```

### 3.1 Ex1：统计未出现的类别

在第五章中介绍了`crosstab`函数，在默认参数下它能够对两个列的组合出现的频数进行统计汇总：


```python
df = pd.DataFrame({'A':['a','b','c','a'], 'B':['cat','cat','dog','cat']})
pd.crosstab(df.A, df.B)
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
      <th>B</th>
      <th>cat</th>
      <th>dog</th>
    </tr>
    <tr>
      <th>A</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



但事实上有些列存储的是分类变量，列中并不一定包含所有的类别，此时如果想要对这些未出现的类别在`crosstab`结果中也进行汇总，则可以指定`dropna`参数为`False`：


```python
df.B = df.B.astype('category').cat.add_categories('sheep')
pd.crosstab(df.A, df.B, dropna=False)
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
      <th>B</th>
      <th>cat</th>
      <th>dog</th>
      <th>sheep</th>
    </tr>
    <tr>
      <th>A</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



请实现一个带有`dropna`参数的`my_crosstab`函数来完成上面的功能。

**我的解答：**


```python
def my_crosstab(s1: pd.Series, s2: pd.Series, dropna=True) -> pd.DataFrame:
    def validate_dropna(s: pd.Series):
        # 如果dropna=False并且series是分类变量
        if s.dtypes.name == 'category' and not dropna:
            return s.cat.categories
        # 否则返回series的不重复数据值作为行列索引
        return s.unique()

    s1_idx = validate_dropna(s1)
    s2_idx = validate_dropna(s2)
    # 构造全0数据
    data = np.zeros((s1_idx.shape[0], s2_idx.shape[0]))
    # 构造DataFrame，行索引为s1中的值，列索引为s2中的值
    res = pd.DataFrame(data, index=s1_idx, columns=s2_idx)
    res.rename_axis(index=s1.name, columns=s2.name, inplace=True)
    # 计算频数
    for s1_idx_value, s2_idx_value in zip(s1, s2):
        res.loc[s1_idx_value, s2_idx_value] += 1
    # 设置行索引
    res = res.astype(np.int64)
    return res
```


```python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'a'], 'B': ['cat', 'cat', 'dog', 'cat']})


def test1():
    res_my_crosstab_df = my_crosstab(df.A, df.B)
    res_crosstab_df = pd.crosstab(df.A, df.B)
    print(res_my_crosstab_df.equals(res_crosstab_df))


def test2():
    df.B = df.B.astype('category').cat.add_categories('sheep')
    res_my_crosstab_df = my_crosstab(df.A, df.B, dropna=False)
    res_crosstab_df = pd.crosstab(df.A, df.B, dropna=False)
    print(res_my_crosstab_df.equals(res_crosstab_df))


test1()
test2()
```

    True
    True
    

### 3.2 Ex2：钻石数据集

现有一份关于钻石的数据集，其中`carat, cut, clarity, price`分别表示克拉重量、切割质量、纯净度和价格，样例如下：


```python
df = pd.read_csv('../data/diamonds.csv') 
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
      <th>carat</th>
      <th>cut</th>
      <th>clarity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>Ideal</td>
      <td>SI2</td>
      <td>326</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>Premium</td>
      <td>SI1</td>
      <td>326</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>Good</td>
      <td>VS1</td>
      <td>327</td>
    </tr>
  </tbody>
</table>
</div>



1. 分别对`df.cut`在`object`类型和`category`类型下使用`nunique`函数，并比较它们的性能。
2. 钻石的切割质量可以分为五个等级，由次到好分别是`Fair, Good, Very Good, Premium, Ideal`，纯净度有八个等级，由次到好分别是`I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF`，请对切割质量按照**由好到次**的顺序排序，相同切割质量的钻石，按照纯净度进行**由次到好**的排序。
3. 分别采用两种不同的方法，把`cut, clarity`这两列按照**由好到次**的顺序，映射到从0到n-1的整数，其中n表示类别的个数。
4. 对每克拉的价格按照分别按照分位数（q=\[0.2, 0.4, 0.6, 0.8\]）与\[1000, 3500, 5500, 18000\]割点进行分箱得到五个类别`Very Low, Low, Mid, High, Very High`，并把按这两种分箱方法得到的`category`序列依次添加到原表中。
5. 第4问中按照整数分箱得到的序列中，是否出现了所有的类别？如果存在没有出现的类别请把该类别删除。
6. 对第4问中按照分位数分箱得到的序列，求每个样本对应所在区间的左右端点值和长度。

**我的解答：**

**第1问：**


```python
s_obj = df.cut
%timeit -n 100 s_obj.nunique()
```

    100 loops, best of 5: 6.39 ms per loop
    


```python
s_cat = df.cut.astype('category')
%timeit -n 100 s_cat.nunique()
```

    100 loops, best of 5: 2.18 ms per loop
    

比较结果：`df.cut`在`category`类型下执行效率更快，比`object`类型快5倍左右

**第2问：**


```python
# 先设置分类变量，并赋予相应的大小关系
cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
df.cut = df.cut.astype('category').cat.reorder_categories(cut_categories, ordered=True)

clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
df.clarity = df.clarity.astype('category').cat.reorder_categories(clarity_categories, ordered=True)
```


```python
# cut 由好到次，故ascending=False
# clarity 由次到好，故ascending=True
res_df = df.sort_values(['cut', 'clarity'], ascending=[False, True])
```


```python
res_df.head()
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
      <th>carat</th>
      <th>cut</th>
      <th>clarity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>315</th>
      <td>0.96</td>
      <td>Ideal</td>
      <td>I1</td>
      <td>2801</td>
    </tr>
    <tr>
      <th>535</th>
      <td>0.96</td>
      <td>Ideal</td>
      <td>I1</td>
      <td>2826</td>
    </tr>
    <tr>
      <th>551</th>
      <td>0.97</td>
      <td>Ideal</td>
      <td>I1</td>
      <td>2830</td>
    </tr>
    <tr>
      <th>653</th>
      <td>1.01</td>
      <td>Ideal</td>
      <td>I1</td>
      <td>2844</td>
    </tr>
    <tr>
      <th>718</th>
      <td>0.97</td>
      <td>Ideal</td>
      <td>I1</td>
      <td>2856</td>
    </tr>
  </tbody>
</table>
</div>



**第3问：**


```python
# cut 由好到次
df.cut = df.cut.cat.reorder_categories(cut_categories[::-1])
```


```python
df.cut.head()
```




    0      Ideal
    1    Premium
    2       Good
    3    Premium
    4       Good
    Name: cut, dtype: category
    Categories (5, object): ['Ideal' < 'Premium' < 'Very Good' < 'Good' < 'Fair']




```python
# clarity 由好到次
df.clarity = df.clarity.cat.reorder_categories(clarity_categories[::-1])
```


```python
df.clarity.head()
```




    0    SI2
    1    SI1
    2    VS1
    3    VS2
    4    SI2
    Name: clarity, dtype: category
    Categories (8, object): ['IF' < 'VVS1' < 'VVS2' < 'VS1' < 'VS2' < 'SI1' < 'SI2' < 'I1']




```python
# 方法一：采用cat.codes进行编号
df.cut = df.cut.cat.codes
df.cut.head()
```




    0    0
    1    1
    2    3
    3    1
    4    3
    Name: cut, dtype: int8




```python
# 方法二：采用cat.rename_categories的方式，构建dict对象
clarity_codes = dict(zip(clarity_categories[::-1], np.arange(len(clarity_categories))))
clarity_codes
```




    {'I1': 7,
     'IF': 0,
     'SI1': 5,
     'SI2': 6,
     'VS1': 3,
     'VS2': 4,
     'VVS1': 1,
     'VVS2': 2}




```python
df.clarity = df.clarity.cat.rename_categories(clarity_codes)
df.clarity.head()
```




    0    6
    1    5
    2    3
    3    4
    4    6
    Name: clarity, dtype: category
    Categories (8, int64): [0 < 1 < 2 < 3 < 4 < 5 < 6 < 7]




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
      <th>carat</th>
      <th>cut</th>
      <th>clarity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>0</td>
      <td>6</td>
      <td>326</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>1</td>
      <td>5</td>
      <td>326</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>3</td>
      <td>3</td>
      <td>327</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>1</td>
      <td>4</td>
      <td>334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>3</td>
      <td>6</td>
      <td>335</td>
    </tr>
  </tbody>
</table>
</div>



**第4问：**


```python
q = [0, 0.2, 0.4, 0.6, 0.8, 1]

bins = [-np.infty, 1000, 3500, 5500, 18000, np.infty]
```


```python
# 每克拉的价格
price_per_carat = df.price / df.carat
```


```python
df['price_per_cut'] = pd.cut(price_per_carat, bins=bins, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
df['price_per_cut'].head()
```




    0    Low
    1    Low
    2    Low
    3    Low
    4    Low
    Name: price_per_cut, dtype: category
    Categories (5, object): ['Very Low' < 'Low' < 'Mid' < 'High' < 'Very High']




```python
df['price_per_qcut'] = pd.qcut(price_per_carat, q=q, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
df['price_per_qcut'].head()
```




    0    Very Low
    1    Very Low
    2    Very Low
    3    Very Low
    4    Very Low
    Name: price_per_qcut, dtype: category
    Categories (5, object): ['Very Low' < 'Low' < 'Mid' < 'High' < 'Very High']




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
      <th>carat</th>
      <th>cut</th>
      <th>clarity</th>
      <th>price</th>
      <th>price_per_cut</th>
      <th>price_per_qcut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>0</td>
      <td>6</td>
      <td>326</td>
      <td>Low</td>
      <td>Very Low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>1</td>
      <td>5</td>
      <td>326</td>
      <td>Low</td>
      <td>Very Low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>3</td>
      <td>3</td>
      <td>327</td>
      <td>Low</td>
      <td>Very Low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>1</td>
      <td>4</td>
      <td>334</td>
      <td>Low</td>
      <td>Very Low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>3</td>
      <td>6</td>
      <td>335</td>
      <td>Low</td>
      <td>Very Low</td>
    </tr>
  </tbody>
</table>
</div>



**第5问：**

根据题意，按照整数分箱，即采用了`pd.cut`方法得到的`price_per_cut`列的数据


```python
df.price_per_cut.unique()
```




    ['Low', 'Mid', 'High']
    Categories (3, object): ['Low' < 'Mid' < 'High']




```python
df.price_per_cut.cat.categories
```




    Index(['Very Low', 'Low', 'Mid', 'High', 'Very High'], dtype='object')



可知有两个分类（`Very Low`, `Very High`）没有出现，删除这两个分类变量


```python
df.price_per_cut = df.price_per_cut.cat.remove_unused_categories()
df.price_per_cut.cat.categories
```




    Index(['Low', 'Mid', 'High'], dtype='object')



**第6问：**

根据题意，按照分位数分箱，即采用了`pd.qcut`方法得到的`price_per_qcut`列的数据


```python
price_per_interval = pd.IntervalIndex(pd.qcut(price_per_carat, q=q))
price_per_interval[:5]
```




    IntervalIndex([(1051.162, 2295.0], (1051.162, 2295.0], (1051.162, 2295.0], (1051.162, 2295.0], (1051.162, 2295.0]],
                  closed='right',
                  dtype='interval[float64]')




```python
# 左端点值
price_per_interval.left.to_series().reset_index(drop=True).head()
```




    0    1051.162
    1    1051.162
    2    1051.162
    3    1051.162
    4    1051.162
    dtype: float64




```python
# 右端点值
price_per_interval.right.to_series().reset_index(drop=True).head()
```




    0    2295.0
    1    2295.0
    2    2295.0
    3    2295.0
    4    2295.0
    dtype: float64




```python
# 长度
price_per_interval.length.to_series().reset_index(drop=True).head()
```




    0    1243.838
    1    1243.838
    2    1243.838
    3    1243.838
    4    1243.838
    dtype: float64


