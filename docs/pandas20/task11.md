# Task11 综合练习 {docsify-ignore-all}


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
| 2018-01-01 00&#58;00&#58;00 | 1 | 1.0 |
| 2018-01-01 00&#58;00&#58;00 | 2 | 1.0 |
| ... | ... | ... |
| 2018-01-01 00&#58;00&#58;00 | 30 | 1.0 |
| 2018-01-01 01&#58;00&#58;00 | 1 | 1.0 |
| 2018-01-01 01&#58;00&#58;00 | 2 | 1.0
| ... | ... | ... |
| 2019-12-31 23&#58;00&#58;00 | 30 | 1.0 |

2. 在上一问构造的`df`基础上，构造下面的特征序列或`DataFrame`，并把它们逐个拼接到`df`的右侧
- 当天最高温、最低温和它们的温差
- 当天是否有沙暴、是否有雾、是否有雨、是否有雪、是否为晴天
- 选择一种合适的方法度量雨量/下雪量的大小（构造两个序列分别表示二者大小）
- 限制只用4列，对风向进行0-1编码

3. 对`df`的水压一列构造如下时序特征：
- 当前时刻该站点水压与本月的相同整点时间水压均值的差，例如当前时刻为2018-05-20 17&#58;00&#58;00，那么对应需要减去的值为当前月所有17&#58;00&#58;00时间点水压值的均值
- 当前时刻所在周的周末该站点水压均值与工作日水压均值之差
- 当前时刻向前7日内，该站点水压的均值、标准差、0.95分位数、下雨天数与下雪天数的总和
- 当前时刻向前7日内，该站点同一整点时间水压的均值、标准差、0.95分位数
- 当前时刻所在日的该站点水压最高值与最低值出现时刻的时间差

【数据下载】  
链接：https://pan.baidu.com/s/1Tqad4b7zN1HBbc-4t4xc6w   
提取码：ijbd

**解答：**


```python
df_2018 = pd.read_csv('../data/task11/task05/yali18.csv')
df_2019 = pd.read_csv('../data/task11/task05/yali19.csv')
df3 = pd.read_csv('../data/task11/task05/qx1819.csv')
```

**第1问：**


```python
def convert_df(df):
    # 先把宽表变长表
    df = df.melt(id_vars=['Time', 'MeasName'],
            value_vars = ['H' + str(i) for i in range(24)],
            var_name = 'Hour',
            value_name = 'HydraulicPressure')
    # 提取小时数
    s_hours = df['Hour'].str.extract('H(\d+)')[0].astype('int64')
    # 利用时间差，得到时序数据
    df['Time'] = pd.to_datetime(df['Time']) + pd.to_timedelta(s_hours, unit='h')
    # 提取站点数
    df['MeasName'] = df['MeasName'].str.extract('站点(\d+)')[0].astype('int64')
    # 转换HydraulicPressure的数据类型
    df['HydraulicPressure'] = pd.to_numeric(df['HydraulicPressure'])
    # 删除Hour列
    df.drop(['Hour'], axis=1, inplace=True) 
    # 排序
    df.sort_values(['Time', 'MeasName'], ignore_index=True, inplace=True)
    return df
```


```python
df_2018 = convert_df(df_2018)
df_2019 = convert_df(df_2019)
```


```python
# 将两个数据集进行拼接
df = pd.concat([df_2018, df_2019])
# 设置Time为索引列
df.set_index(['Time'], inplace=True)
df.index.name = ''
```


```python
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
      <th>MeasName</th>
      <th>HydraulicPressure</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00&#58;00&#58;00</th>
      <td>1</td>
      <td>0.288625</td>
    </tr>
    <tr>
      <th>2018-01-01 00&#58;00&#58;00</th>
      <td>2</td>
      <td>0.317750</td>
    </tr>
    <tr>
      <th>2018-01-01 00&#58;00&#58;00</th>
      <td>3</td>
      <td>0.301375</td>
    </tr>
    <tr>
      <th>2018-01-01 00&#58;00&#58;00</th>
      <td>4</td>
      <td>0.402750</td>
    </tr>
    <tr>
      <th>2018-01-01 00&#58;00&#58;00</th>
      <td>5</td>
      <td>0.314625</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-12-31 23&#58;00&#58;00</th>
      <td>26</td>
      <td>0.323250</td>
    </tr>
    <tr>
      <th>2019-12-31 23&#58;00&#58;00</th>
      <td>27</td>
      <td>0.312000</td>
    </tr>
    <tr>
      <th>2019-12-31 23&#58;00&#58;00</th>
      <td>28</td>
      <td>0.294500</td>
    </tr>
    <tr>
      <th>2019-12-31 23&#58;00&#58;00</th>
      <td>29</td>
      <td>0.265875</td>
    </tr>
    <tr>
      <th>2019-12-31 23&#58;00&#58;00</th>
      <td>30</td>
      <td>0.274875</td>
    </tr>
  </tbody>
</table>
<p>525600 rows × 2 columns</p>
</div>



**第2问：**


```python
df3 = pd.read_csv('../data/task11/task05/qx1819.csv')
df3['日期'] = pd.to_datetime(df3['日期'])
```


```python
df_q21 = df3.copy()

res = df_q21['气温'].str.extract('(?P<One>-?\d+)[C|℃]\s{0,2}～\s{0,2}(?P<Two>-?\d+)[C|℃]')
res['One'] = pd.to_numeric(res.One)
res['Two'] = pd.to_numeric(res.Two)
# 最高温，最低温
df_q21['High_Temp'] = res.max(1)
df_q21['Low_Temp'] = res.min(1)

# 对数据进行修正
df_q21.iloc[22, -2] = df_q21[df_q21.Low_Temp == -5].High_Temp.mean()
df_q21.iloc[22, -1] = -5
df_q21.iloc[643, -1] = df_q21[df_q21.High_Temp == 9].Low_Temp.mean()
df_q21.iloc[643, -2] = 9

# 温差
df_q21['Delta_Temp'] = df_q21['High_Temp'] - df_q21['Low_Temp']
```


```python
df_q21 = df_q21[['日期', 'High_Temp', 'Low_Temp', 'Delta_Temp']]

# 拼接到df的右侧
def concat_df(df, res):
    res = res.set_index('日期').reindex(pd.date_range('20180101', '20191231 23:00:00', freq='H')).fillna(method='ffill')
    return df.join(res, how='left')
    
res = concat_df(df, df_q21)
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
      <th>MeasName</th>
      <th>HydraulicPressure</th>
      <th>High_Temp</th>
      <th>Low_Temp</th>
      <th>Delta_Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>1</td>
      <td>0.288625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>2</td>
      <td>0.317750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>3</td>
      <td>0.301375</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>4</td>
      <td>0.402750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>5</td>
      <td>0.314625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_q22 = df3.copy()

my_dict = dict(zip(['沙', '雾', '雨', '雪', '晴'], ['Sand', 'Fog', 'Rain', 'Snow', 'Sun']))

for i in my_dict.keys():
    df_q22[my_dict[i]] = df_q22['天气'].str.contains(i).astype('int64')
    
df_q22 = df_q22[['日期', 'Sand', 'Fog', 'Rain', 'Snow', 'Sun']]

res = concat_df(res, df_q22)
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
      <th>MeasName</th>
      <th>HydraulicPressure</th>
      <th>High_Temp</th>
      <th>Low_Temp</th>
      <th>Delta_Temp</th>
      <th>Sand</th>
      <th>Fog</th>
      <th>Rain</th>
      <th>Snow</th>
      <th>Sun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>1</td>
      <td>0.288625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>2</td>
      <td>0.317750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>3</td>
      <td>0.301375</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>4</td>
      <td>0.402750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>5</td>
      <td>0.314625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_q23 = df3.copy()

my_dict = dict(zip(['大雨', '中雨', '小雨'], [3, 2, 1]))
df_q23['Rain_Range'] = 0

for i in my_dict.keys():
    df_q23.loc[df_q23['天气'].str.contains(i), 'Rain_Range'] = my_dict[i]
    
res1 = df_q23[['日期', 'Rain_Range']]
res1.head()
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
      <th>日期</th>
      <th>Rain_Range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_dict = dict(zip(['大雪', '中雪', '小雪'], [3, 2, 1]))
df_q23['Sown_Range'] = 0

for i in my_dict.keys():
    df_q23.loc[df_q23['天气'].str.contains(i), 'Sown_Range'] = my_dict[i]
    
res2 = df_q23[['日期', 'Sown_Range']]
res2.head()
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
      <th>日期</th>
      <th>Sown_Range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
res_q23 = pd.concat([res1, res2['Sown_Range']], 1)
res_q23.head()
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
      <th>日期</th>
      <th>Rain_Range</th>
      <th>Sown_Range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = concat_df(res, res_q23)
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
      <th>MeasName</th>
      <th>HydraulicPressure</th>
      <th>High_Temp</th>
      <th>Low_Temp</th>
      <th>Delta_Temp</th>
      <th>Sand</th>
      <th>Fog</th>
      <th>Rain</th>
      <th>Snow</th>
      <th>Sun</th>
      <th>Rain_Range</th>
      <th>Sown_Range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>1</td>
      <td>0.288625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>2</td>
      <td>0.317750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>3</td>
      <td>0.301375</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>4</td>
      <td>0.402750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>5</td>
      <td>0.314625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in ['Sand', 'Fog', 'Rain', 'Snow', 'Sun', 'Rain_Range', 'Sown_Range']:
    res[i] = res[i].astype('int64')
```


```python
df_q24 = df3.copy()

df_q24 = pd.concat([df_q24, 
                 pd.concat([df_q24['风向'].str.contains(i).rename(i).astype('int64') for i in list('东南西北')], 1)], 1)

df_q24 = df_q24[['日期', *list('东南西北')]]
df_q24.head()
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
      <th>日期</th>
      <th>东</th>
      <th>南</th>
      <th>西</th>
      <th>北</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
res = concat_df(res, df_q24)

for i in ['东','南', '西', '北']:
    res[i] = res[i].astype('int64')

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
      <th>MeasName</th>
      <th>HydraulicPressure</th>
      <th>High_Temp</th>
      <th>Low_Temp</th>
      <th>Delta_Temp</th>
      <th>Sand</th>
      <th>Fog</th>
      <th>Rain</th>
      <th>Snow</th>
      <th>Sun</th>
      <th>Rain_Range</th>
      <th>Sown_Range</th>
      <th>东</th>
      <th>南</th>
      <th>西</th>
      <th>北</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>1</td>
      <td>0.288625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>2</td>
      <td>0.317750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>3</td>
      <td>0.301375</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>4</td>
      <td>0.402750</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>5</td>
      <td>0.314625</td>
      <td>1.0</td>
      <td>-4.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**第3问：**


```python
def fun1(x):
    res = x.groupby([x.index.year, x.index.month, x.index.hour]).transform(
        lambda x: x-x.mean())
    return res

res_q31 = res.groupby('MeasName')['HydraulicPressure'].transform(fun1)
res_q31
```




    2018-01-01 00:00:00    0.014988
    2018-01-01 00:00:00    0.000835
    2018-01-01 00:00:00    0.000738
    2018-01-01 00:00:00   -0.015472
    2018-01-01 00:00:00    0.001440
                             ...   
    2019-12-31 23:00:00   -0.004948
    2019-12-31 23:00:00   -0.021786
    2019-12-31 23:00:00   -0.009992
    2019-12-31 23:00:00   -0.012544
    2019-12-31 23:00:00   -0.011625
    Name: HydraulicPressure, Length: 525600, dtype: float64




```python
def fun2(x):
    temp = x.index.dayofweek.isin([0,1,2,3,4])
    weekday = x[temp].mean()
    weekend = x[~temp].mean()
    return weekend - weekday

res_q32 = res.groupby('MeasName')['HydraulicPressure'].resample('w').agg(fun2)
res_q32
```




    MeasName            
    1         2018-01-07   -0.001511
              2018-01-14   -0.003805
              2018-01-21   -0.008898
              2018-01-28   -0.000592
              2018-02-04   -0.006963
                              ...   
    30        2019-12-08   -0.002114
              2019-12-15   -0.004691
              2019-12-22   -0.000778
              2019-12-29   -0.005813
              2020-01-05         NaN
    Name: HydraulicPressure, Length: 3150, dtype: float64




```python
res.groupby('MeasName')['HydraulicPressure'].transform(lambda x: x.rolling('7D').mean())
```




    2018-01-01 00:00:00    0.288625
    2018-01-01 00:00:00    0.317750
    2018-01-01 00:00:00    0.301375
    2018-01-01 00:00:00    0.402750
    2018-01-01 00:00:00    0.314625
                             ...   
    2019-12-31 23:00:00    0.321609
    2019-12-31 23:00:00    0.331016
    2019-12-31 23:00:00    0.297737
    2019-12-31 23:00:00    0.271694
    2019-12-31 23:00:00    0.276750
    Name: HydraulicPressure, Length: 525600, dtype: float64




```python
def fun4(x):
    res = x.rolling(24*7+1).apply(lambda x:x.iloc[0::24].mean())
    return res

res.groupby('MeasName')['HydraulicPressure'].transform(fun4)
```




    2018-01-01 00:00:00         NaN
    2018-01-01 00:00:00         NaN
    2018-01-01 00:00:00         NaN
    2018-01-01 00:00:00         NaN
    2018-01-01 00:00:00         NaN
                             ...   
    2019-12-31 23:00:00    0.327094
    2019-12-31 23:00:00    0.332344
    2019-12-31 23:00:00    0.302281
    2019-12-31 23:00:00    0.278250
    2019-12-31 23:00:00    0.283453
    Name: HydraulicPressure, Length: 525600, dtype: float64




```python
res.groupby(['MeasName', df.index.date])['HydraulicPressure'].agg(
    lambda x: x.idxmax() - x.idxmin())
```




    MeasName            
    1         2018-01-01   -1 days +18:00:00
              2018-01-02   -1 days +18:00:00
              2018-01-03   -1 days +06:00:00
              2018-01-04   -1 days +17:00:00
              2018-01-05   -1 days +16:00:00
                                  ...       
    30        2019-12-27   -1 days +19:00:00
              2019-12-28   -1 days +18:00:00
              2019-12-29   -1 days +07:00:00
              2019-12-30   -1 days +16:00:00
              2019-12-31   -1 days +07:00:00
    Name: HydraulicPressure, Length: 21900, dtype: timedelta64[ns]


