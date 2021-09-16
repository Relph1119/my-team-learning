# Task03 布局格式定方圆


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号
```

## 1 知识梳理

### 1.1 均匀子图

- 使用plt.subplots
  1. 返回元素：画布figure，子图axes列表
  2. 常用参数：
      - 第1,2个参数：行列个数
      - figsize：指定整个画布的大小
      - sharex：共享横轴刻度
      - sharey：共享纵轴刻度
      - projection：`polar`表示极坐标
  3. 方法：
      - tight_layout：调整子图的相对大小，使字符不会重复

### 1.2 非均匀子图

- 含义：
  1. 图的比例大小不同，但没有跨行或跨列
  2. 图为跨行或跨列状态
- 使用add_gridspec
  1. 常用参数：
      - width_ratios：相对宽度比例
      - height_ratios：相对高度比例

### 1.3 子图上的方法

- plot：绘制折线
- hist：绘制直方图
- axhline：绘制水平直线
- axvline：绘制垂直直线
- axline：绘制任意方向直线
- set_xscale：设置横坐标轴的规度（指对数坐标等）
- set_title：设置标题
- set_xlabel：设置轴名

- legend：绘制图例
- annotate：绘制注释
- arrow：绘制带箭头的直线
- text：绘制文字

- 图例的 `loc` 参数如下：

|  string   | code  |
|  ----  | ----  |
| best  | 0 |
| upper right  | 1 |
| upper left  | 2 |
| lower left  | 3 |
| lower right  | 4 |
| right  | 5 |
| center left  | 6 |
| center right  | 7 |
| lower center  | 8 |
| upper center  | 9 |
| center  | 10 |

## 2 实战练习

### 2.1 绘制均匀子图


```python
# 创建2行5列的均匀子图
fig, axs = plt.subplots(2, 5, figsize=(10, 4), sharex=True, sharey=True)
# 创建标题
fig.suptitle('样例1', size=20)

for i in range(2):
    for j in range(5):
        # 绘制散点图
        axs[i][j].scatter(np.random.randn(10), np.random.randn(10))
        axs[i][j].set_title(f'第{i+1}行，第{j+1}列')
        axs[i][j].set_xlim(-5, 5)
        axs[i][j].set_ylim(-5, 5)
        if i == 1:
            axs[i][j].set_xlabel('横坐标')
        if j == 0:
            axs[i][j].set_ylabel('纵坐标')

# 自动调整子图的相对大小
fig.tight_layout()
```


    
![png](images/task03/output_13_0.png)
    


### 2.2 绘制非均匀子图


```python
fig = plt.figure(figsize=(10, 4))
# 创建非均匀子图列，相对宽度比例是1:2:3:4:5，相对高度比例是1:3
spec = fig.add_gridspec(nrows=2, ncols=5, width_ratios=[
                        1, 2, 3, 4, 5], height_ratios=[1, 3])
# 创建标题
fig.suptitle('样例2', size=20)
for i in range(2):
    for j in range(5):
        # 添加子图
        ax = fig.add_subplot(spec[i, j])
        # 绘制散点图
        ax.scatter(np.random.randn(10), np.random.randn(10))
        ax.set_title(f'第{i+1}行，第{j+1}列')
        if i == 1:
            ax.set_xlabel('横坐标')
        if j == 0:
            ax.set_ylabel('纵坐标')

# 自动调整子图的相对大小
fig.tight_layout()
```


    
![png](images/task03/output_15_0.png)
    



```python
fig = plt.figure(figsize=(10, 4))
# 创建非均匀子图列2行6列，相对宽度比例是2:2.5:3:1:1.5:2，相对高度比例是1:2
spec = fig.add_gridspec(nrows=2, ncols=6, width_ratios=[
                        2, 2.5, 3, 1, 1.5, 2], height_ratios=[1, 2])
fig.suptitle('样例3', size=20)
# 跨列sub1
ax = fig.add_subplot(spec[0, :3])
ax.scatter(np.random.randn(10), np.random.randn(10))
# 跨列sub2
ax = fig.add_subplot(spec[0, 3:5])
ax.scatter(np.random.randn(10), np.random.randn(10))
# 跨行sub3
ax = fig.add_subplot(spec[:, 5])
ax.scatter(np.random.randn(10), np.random.randn(10))
# sub4
ax = fig.add_subplot(spec[1, 0])
ax.scatter(np.random.randn(10), np.random.randn(10))
# 跨列sub5
ax = fig.add_subplot(spec[1, 1:5])
ax.scatter(np.random.randn(10), np.random.randn(10))

# 自动调整子图的相对大小
fig.tight_layout()
```


    
![png](images/task03/output_16_0.png)
    


### 2.3 绘制各种对象


```python
# 创建figure，axes
fig, ax = plt.subplots()
# 绘制带箭头的直线：起点(0,0)，终点(1,1)，箭头宽度0.03，长度0.05，外框颜色blue，内部颜色red
ax.arrow(0, 0, 1, 1, head_width=0.03, head_length=0.05,
         facecolor='red', edgecolor='blue')
# 绘制文字：起点(0,0)，字体16，角度70，文字颜色green
ax.text(x=0, y=0, s='这是一段文字', fontsize=16, rotation=70,
        rotation_mode='anchor', color='green')
# 绘制注释：注释位置(0.5,0.5)，文字位置(0.8,0.2)，箭头颜色yellow，外框颜色black，字体16
ax.annotate('这是中点', xy=(0.5, 0.5), xytext=(0.8, 0.2), arrowprops=dict(
    facecolor='yellow', edgecolor='black'), fontsize=16)
```




    Text(0.8, 0.2, '这是中点')




    
![png](images/task03/output_18_1.png)
    


## 3 课后习题

### 3.1 习题1

墨尔本1981年至1990年的每月温度情况


```python
ex1 = pd.read_csv('../data/layout_ex1.csv')
ex1.head()
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
      <th>Time</th>
      <th>Temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-01</td>
      <td>17.712903</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-02</td>
      <td>17.678571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-03</td>
      <td>13.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-04</td>
      <td>12.356667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-05</td>
      <td>9.490323</td>
    </tr>
  </tbody>
</table>
</div>



请利用数据，画出如下的图：

<img src="https://s1.ax1x.com/2020/11/01/BwvCse.png" width="800" align="bottom" />

**解答：**


```python
# 将Time列进行拆分，分成year和month两列
new_cols = ex1['Time'].str.split('-', expand=True).rename(columns={0:'year', 1:'month'})
# 将数据进行整合
ex1 = pd.concat([ex1.drop(columns=['Time']), new_cols], 1)
# 修改year和month的数据类型
ex1.month = ex1.month.astype('int').astype('str')
ex1.head()
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
      <th>Temperature</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.712903</td>
      <td>1981</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17.678571</td>
      <td>1981</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.500000</td>
      <td>1981</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.356667</td>
      <td>1981</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.490323</td>
      <td>1981</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
years = ex1['year'].unique().tolist()
months = ex1['month'].unique()
```


```python
fig, axs = plt.subplots(2, 5, figsize=(16, 4), sharex=True, sharey=True)
fig.suptitle('墨尔本1981年至1990年月温度曲线', size=16)
year_index = 0
for i in range(2):
    for j in range(5):
        year = str(years[year_index])
        year_index += 1
        axs[i][j].plot(months, ex1.loc[ex1.year==year]['Temperature'])
        axs[i][j].scatter(months, ex1.loc[ex1.year==year]['Temperature'], s=12)
        axs[i][j].set_title(f'{year}年')
        if j==0: axs[i][j].set_ylabel('气温')
        
fig.tight_layout()
```


    
![png](images/task03/output_27_0.png)
    


### 3.2 习题2

画出数据的散点图和边际分布，使用 `np.random.randn(2, 150)` 生成一组二维数据，使用两种非均匀子图的分割方法，做出该数据对应的散点图和边际分布图

<img src="https://s1.ax1x.com/2020/11/01/B0pEnS.png" width="400" height="400" align="bottom" />

**解答：**

**方法1：**使用`add_gridspec`方式


```python
data = np.random.randn(2, 150)
np.random.seed(15)

fig = plt.figure(figsize=(6, 6))

# 创建非均匀子图2行2列，相对宽度比例是5:1，相对高度比例是1:6
spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[5, 1] ,height_ratios=[1,6])

for i in range(2):
    for j in range(2):
        if i == j:
            ax = fig.add_subplot(spec[i, j], frame_on=False, xticks=[], yticks=[])
            if i == 0:
                ax.hist(data[0], bins=10, rwidth=0.9)
            elif i == 1:
                ax.hist(data[1], bins=10, orientation = 'horizontal',rwidth=0.9)
        elif i == 1 and j == 0:
            ax = fig.add_subplot(spec[i, j], xlabel='my_data_x', ylabel='my_data_y')
            ax.scatter(data[0], data[1])
            ax.grid(True)

# 自动调整子图的相对大小
fig.tight_layout()
```


    
![png](images/task03/output_33_0.png)
    


**方法2：**通过切片实现子图


```python
fig = plt.figure(figsize=(6, 6))
spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[5, 1] ,height_ratios=[1,6])

# 绘制data_x的边际分布图
ax = fig.add_subplot(spec[0, 0], frame_on=False, xticks=[], yticks=[])
ax.hist(data[0], bins=10, rwidth=0.9)

# 绘制散点图
ax = fig.add_subplot(spec[1, 0], xlabel='my_data_x', ylabel='my_data_y')
ax.scatter(data[0], data[1])
ax.grid(True)

# 绘制data_y的边际分布图
ax = fig.add_subplot(spec[1, 1], frame_on=False, xticks=[], yticks=[])
ax.hist(data[1], bins=10, orientation = 'horizontal',rwidth=0.9)

fig.tight_layout()
```


    
![png](images/task03/output_35_0.png)
    


## 4 总结

&emsp;&emsp;本次任务，主要介绍了绘制均匀子图和非均匀子图，并讲解了Axes对象的图形绘制函数，通过习题，熟悉了子图、散点图和直方图的绘制。
