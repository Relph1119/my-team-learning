# Task11 综合练习


```python
import pandas as pd
import numpy as np
```

## 任务四 显卡日志

【题目描述】  
下面给出了3090显卡的性能测评日志结果，每一条日志有如下结构：
```
Benchmarking #2# #4# precision type #1#  
#1#  model average #2# time :  #3# ms
```
其中：
- #1#代表的是模型名称
- #2#的值为train(ing)或inference，表示训练状态或推断状态
- #3#表示耗时
- #4#表示精度，其中包含了float, half, double三种类型，

下面是一个具体的例子：
```
Benchmarking Inference float precision type resnet50
resnet50  model average inference time :  13.426570892333984 ms
```
请把日志结果进行整理，变换成如下状态，`model_i`用相应模型名称填充，按照字母顺序排序，数值保留三位小数：

| |Train_half | Train_float |	Train_double | Inference_half | Inference_float	| Inference_double |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| model_1 | 0.954 | 0.901 | 0.357 | 0.281 | 0.978 | 1.130 |
| model_2 | 0.360 | 0.794 | 0.011 | 1.083 | 1.137 | 0.394 |
| … | … | …	| … | …	| … | … 

【数据下载】  
链接：https://pan.baidu.com/s/1CjfdtavEywHtZeWSmCGv3A  
提取码：4mui

**解答：**


```python
# 将日志内容存储到lines里
lines = []
for line in open('../data/task11/task04/benchmark.txt'): 
    line = line.replace('\n', '')
    if 'Benchmarking' in line and 'precision type' in line:
        lines.append(line)
    if 'model average' in line:
        lines.append(line)
```


```python
# 构造df
df = pd.DataFrame(columns=['model', 'state', 'precision_type', 'time'])
i = 0
while i < len(lines):
    line1 = lines[i]
    s1 = pd.Series(line1)
    pat1 = 'Benchmarking (?P<state>Training|Inference) (?P<precision_type>float|half|double) precision type (?P<model>\w+)'
    first = s1.str.extract(pat1).iloc[0,:]
    line2 = lines[i+1]
    s2 = pd.Series(line2)
    pat2 = '(?P<model>\w+)  model average (?P<state>train|inference) time :  (?P<time>.+) ms'
    second = s2.str.extract(pat2).iloc[0,:]
    s = pd.Series(data=[first['model'], first['state'], first['precision_type'], second['time']], 
                  index=['model', 'state', 'precision_type', 'time'])
    df = df.append(s, ignore_index=True)
    i += 2    
```


```python
# 对耗时的数值保留三位小数
df.time = df.time.astype(np.float64).round(3)
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
      <th>model</th>
      <th>state</th>
      <th>precision_type</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mnasnet0_5</td>
      <td>Training</td>
      <td>float</td>
      <td>28.528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mnasnet0_75</td>
      <td>Training</td>
      <td>float</td>
      <td>34.105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mnasnet1_0</td>
      <td>Training</td>
      <td>float</td>
      <td>34.314</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mnasnet1_3</td>
      <td>Training</td>
      <td>float</td>
      <td>35.557</td>
    </tr>
    <tr>
      <th>4</th>
      <td>resnet18</td>
      <td>Training</td>
      <td>float</td>
      <td>18.660</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 进行长表变宽表
df = df.pivot(index='model', columns=['state', 'precision_type'], values='time')
df = df.sort_index(axis = 1).rename(columns={'Training':'Train'}, level=0)
# 将列名进行合并
df.columns = df.columns.map(lambda x: (x[0] + '_' + x[1]))
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
      <th>Inference_double</th>
      <th>Inference_float</th>
      <th>Inference_half</th>
      <th>Train_double</th>
      <th>Train_float</th>
      <th>Train_half</th>
    </tr>
    <tr>
      <th>model</th>
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
      <th>densenet121</th>
      <td>144.111</td>
      <td>15.637</td>
      <td>19.772</td>
      <td>417.207</td>
      <td>93.357</td>
      <td>88.976</td>
    </tr>
    <tr>
      <th>densenet161</th>
      <td>511.177</td>
      <td>31.750</td>
      <td>27.555</td>
      <td>1290.287</td>
      <td>136.624</td>
      <td>144.319</td>
    </tr>
    <tr>
      <th>densenet169</th>
      <td>175.808</td>
      <td>21.598</td>
      <td>26.371</td>
      <td>511.404</td>
      <td>104.840</td>
      <td>121.556</td>
    </tr>
    <tr>
      <th>densenet201</th>
      <td>223.960</td>
      <td>26.169</td>
      <td>33.394</td>
      <td>654.365</td>
      <td>129.334</td>
      <td>118.940</td>
    </tr>
    <tr>
      <th>mnasnet0_5</th>
      <td>11.870</td>
      <td>8.039</td>
      <td>6.929</td>
      <td>48.232</td>
      <td>28.528</td>
      <td>27.198</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 修改列名顺序
new_columns = ['Train_half', 'Train_float', 'Train_double', 'Inference_half',
               'Inference_float', 'Inference_double']
df = df.reindex(columns=new_columns).reset_index()
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
      <th>model</th>
      <th>Train_half</th>
      <th>Train_float</th>
      <th>Train_double</th>
      <th>Inference_half</th>
      <th>Inference_float</th>
      <th>Inference_double</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>densenet121</td>
      <td>88.976</td>
      <td>93.357</td>
      <td>417.207</td>
      <td>19.772</td>
      <td>15.637</td>
      <td>144.111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>densenet161</td>
      <td>144.319</td>
      <td>136.624</td>
      <td>1290.287</td>
      <td>27.555</td>
      <td>31.750</td>
      <td>511.177</td>
    </tr>
    <tr>
      <th>2</th>
      <td>densenet169</td>
      <td>121.556</td>
      <td>104.840</td>
      <td>511.404</td>
      <td>26.371</td>
      <td>21.598</td>
      <td>175.808</td>
    </tr>
    <tr>
      <th>3</th>
      <td>densenet201</td>
      <td>118.940</td>
      <td>129.334</td>
      <td>654.365</td>
      <td>33.394</td>
      <td>26.169</td>
      <td>223.960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mnasnet0_5</td>
      <td>27.198</td>
      <td>28.528</td>
      <td>48.232</td>
      <td>6.929</td>
      <td>8.039</td>
      <td>11.870</td>
    </tr>
  </tbody>
</table>
</div>



## 任务五 水压站点的特征工程

`df1`和`df2`中分别给出了18年和19年各个站点的数据，其中列中的H0至H23分别代表当天0点至23点；`df3`中记录了18-19年的每日该地区的天气情况，请完成如下的任务：

```python
import pandas as pd
import numpy as np
df1 = pd.read_csv('yali18.csv')
df2 = pd.read_csv('yali19.csv')
df3 = pd.read_csv('qx1819.csv')
```

1. 通过`df1`和`df2`构造`df`，把时间设为索引，第一列为站点编号，第二列为对应时刻的压力大小，排列方式如下（压力数值请用正确的值替换）：

| |站点 | 压力 |
|:---:|:---:|:---:|
| 2018-01-01 00:00:00 | 1 | 1.0 |
| 2018-01-01 00:00:00 | 2 | 1.0 |
| ... | ... | ... |
| 2018-01-01 00:00:00 | 30 | 1.0 |
| 2018-01-01 01:00:00 | 1 | 1.0 |
| 2018-01-01 01:00:00 | 2 | 1.0
| ... | ... | ... |
| 2019-12-31 23:00:00 | 30 | 1.0 |

2. 在上一问构造的`df`基础上，构造下面的特征序列或`DataFrame`，并把它们逐个拼接到`df`的右侧
- 当天最高温、最低温和它们的温差
- 当天是否有沙暴、是否有雾、是否有雨、是否有雪、是否为晴天
- 选择一种合适的方法度量雨量/下雪量的大小（构造两个序列分别表示二者大小）
- 限制只用4列，对风向进行0-1编码

3. 对`df`的水压一列构造如下时序特征：
- 当前时刻该站点水压与本月的相同整点时间水压均值的差，例如当前时刻为2018-05-20 17:00:00，那么对应需要减去的值为当前月所有17:00:00时间点水压值的均值
- 当前时刻所在周的周末该站点水压均值与工作日水压均值之差
- 当前时刻向前7日内，该站点水压的均值、标准差、0.95分位数、下雨天数与下雪天数的总和
- 当前时刻向前7日内，该站点同一整点时间水压的均值、标准差、0.95分位数
- 当前时刻所在日的该站点水压最高值与最低值出现时刻的时间差

【数据下载】  
链接：https://pan.baidu.com/s/1Tqad4b7zN1HBbc-4t4xc6w   
提取码：ijbd


```python

```
