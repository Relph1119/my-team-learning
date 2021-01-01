# Task Special 综合练习


```python
import pandas as pd
import numpy as np
```

## 任务一 企业收入的多样性

【题目描述】  
一个企业的产业收入多样性可以仿照信息熵的概念来定义收入熵指标：$$I=-\sum_{i}p(x_i)\log(p(x_i))$$
其中$p(x_i)$是企业该年某产业收入额占该年所有产业总收入的比重。在`company.csv`中存有需要计算的企业和年份，在`company_data.csv`中存有企业、各类收入额和收入年份的信息。现请利用后一张表中的数据，在前一张表中增加一列表示该公司该年份的收入熵指标$I$。

【数据下载】  
链接：https://pan.baidu.com/s/1leZZctxMUSW55kZY5WwgIw   
密码：u6fd

**解答：**


```python
# 读取Company.csv数据
company = pd.read_csv('../data/task_special/task01/Company.csv')
company.columns = ['Code', 'Date']
company.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#000007</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#000403</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#000408</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#000408</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#000426</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 读取Company_data.csv数据
company_data = pd.read_csv('../data/task_special/task01/Company_data.csv')
company_data.columns = ['Code', 'Date', 'Type', 'Amount']
company_data['Date'] = pd.to_datetime(company_data['Date'])
company_data.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Date</th>
      <th>Type</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2008-12-31</td>
      <td>1</td>
      <td>1.084218e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2008-12-31</td>
      <td>2</td>
      <td>1.259789e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2008-12-31</td>
      <td>3</td>
      <td>1.451312e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2008-12-31</td>
      <td>4</td>
      <td>1.063843e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2008-12-31</td>
      <td>5</td>
      <td>8.513880e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 数据清洗
# 1. 将负数取绝对值
company_data['Amount'] = company_data['Amount'].abs()
# 2. 将空值删除
company_data = company_data.dropna(how='any', subset=['Amount'])
# 3. 取出Date列中的年
company_data['Date'] = company_data['Date'].dt.year
```


```python
company_data.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Date</th>
      <th>Type</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2008</td>
      <td>1</td>
      <td>1.084218e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2008</td>
      <td>2</td>
      <td>1.259789e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2008</td>
      <td>3</td>
      <td>1.451312e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2008</td>
      <td>4</td>
      <td>1.063843e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2008</td>
      <td>5</td>
      <td>8.513880e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 得到比重值
company_data['Px'] = company_data.groupby(['Code', 'Date'])['Amount'].apply(lambda x: x/x.sum())
```


```python
# 根据公式计算I值
company_data_tmp = company_data.groupby(['Code', 'Date'])['Px'].agg(lambda x: -sum([p * np.log(p)  for p in x.tolist()])).to_frame()
```

    E:\Learning_Projects\MyPythonProjects\my-team-learning\venv\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: divide by zero encountered in log
      
    E:\Learning_Projects\MyPythonProjects\my-team-learning\venv\lib\site-packages\ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in double_scalars
      
    


```python
# 去掉索引，将Code格式化
company_data_tmp = company_data_tmp.reset_index()
company_data_tmp['Code'] = company_data_tmp['Code'].apply(lambda x: '#{0:0>6}'.format(x))
company_data_tmp.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Code</th>
      <th>Date</th>
      <th>Px</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#000001</td>
      <td>2008</td>
      <td>2.125238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#000001</td>
      <td>2009</td>
      <td>1.671752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#000001</td>
      <td>2010</td>
      <td>2.108355</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#000001</td>
      <td>2011</td>
      <td>3.155371</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#000001</td>
      <td>2012</td>
      <td>2.738493</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 进行连接
res = company.merge(company_data_tmp, on=['Code', 'Date'], how='left')
# 将列重命名
res = res.rename(columns={'Code':'证券代码', 'Date':'日期', 'Px':'收入熵指标'})
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>证券代码</th>
      <th>日期</th>
      <th>收入熵指标</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#000007</td>
      <td>2014</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#000403</td>
      <td>2015</td>
      <td>2.790585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#000408</td>
      <td>2016</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#000408</td>
      <td>2017</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#000426</td>
      <td>2015</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 保存数据
res.to_csv('../data/task_special/task01/task01_result.csv', index=False)
```

## 任务二组队学习信息表的变换

【题目描述】  
请把组队学习的队伍信息表变换为如下形态，其中“是否队长”一列取1表示队长，否则为0

<img src="../source/_static/ch_special.png" width="40%">

【数据下载】   
链接：https://pan.baidu.com/s/1ses24cTwUCbMx3rvYXaz-Q  
密码：iz57

**解答：**


```python
team_data = pd.read_excel('../data/task_special/task02/组队信息汇总表（Pandas）.xls')
team_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21 entries, 0 to 20
    Data columns (total 24 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   所在群       21 non-null     object 
     1   队伍名称      21 non-null     object 
     2   队长编号      21 non-null     int64  
     3   队长_群昵称    21 non-null     object 
     4   队员1 编号    21 non-null     int64  
     5   队员_群昵称    21 non-null     object 
     6   队员2 编号    20 non-null     float64
     7   队员_群昵称.1  20 non-null     object 
     8   队员3 编号    18 non-null     float64
     9   队员_群昵称.2  18 non-null     object 
     10  队员4 编号    16 non-null     float64
     11  队员_群昵称.3  16 non-null     object 
     12  队员5 编号    14 non-null     float64
     13  队员_群昵称.4  14 non-null     object 
     14  队员6 编号    13 non-null     float64
     15  队员_群昵称.5  13 non-null     object 
     16  队员7 编号    10 non-null     float64
     17  队员_群昵称.6  10 non-null     object 
     18  队员8 编号    8 non-null      float64
     19  队员_群昵称.7  8 non-null      object 
     20  队员9 编号    4 non-null      float64
     21  队员_群昵称.8  4 non-null      object 
     22  队员10编号    1 non-null      float64
     23  队员_群昵称.9  1 non-null      object 
    dtypes: float64(9), int64(2), object(13)
    memory usage: 4.1+ KB
    

## 任务三 美国大选投票情况

【题目描述】  
两张数据表中分别给出了美国各县（county）的人口数以及大选的投票情况，请解决以下问题：
- 有多少县满足总投票数超过县人口数的一半
- 把州（state）作为行索引，把投票候选人作为列名，列名的顺序按照候选人在全美的总票数由高到低排序，行列对应的元素为该候选人在该州获得的总票数

    \# 此处是一个样例，实际的州或人名用原表的英语代替

||拜登|川普|
|---|---|---|
|威斯康星州|2|1|
|德克萨斯州|3|4|

- 每一个州下设若干县，定义拜登在该县的得票率减去川普在该县的得票率为该县的BT指标，若某个州所有县BT指标的中位数大于0，则称该州为Biden State，请找出所有的Biden State

【数据下载】  
链接：https://pan.baidu.com/s/182rr3CpstVux2CFdFd_Pcg  
提取码：q674

**解答：**

**第1问：有多少县满足总投票数超过县人口数的一半？**


```python
# 读取county_population.csv
county_population = pd.read_csv('../data/task_special/task03/county_population.csv')
# 分隔US County列
county_population['county'], county_population['state'] = county_population['US County'].str.split(',.', 1).str
county_population['county'] = county_population['county'].str[1:]
# 删除US County列
county_population = county_population.drop(columns='US County')
county_population.head()
```

    E:\Learning_Projects\MyPythonProjects\my-team-learning\venv\lib\site-packages\ipykernel_launcher.py:4: FutureWarning: Columnar iteration over characters will be deprecated in future releases.
      after removing the cwd from sys.path.
    




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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>county</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55869</td>
      <td>Autauga County</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>223234</td>
      <td>Baldwin County</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24686</td>
      <td>Barbour County</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22394</td>
      <td>Bibb County</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57826</td>
      <td>Blount County</td>
      <td>Alabama</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 读取president_county_candidate.csv
president_county_candidate = pd.read_csv('../data/task_special/task03/president_county_candidate.csv')
president_county_candidate.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>county</th>
      <th>candidate</th>
      <th>party</th>
      <th>total_votes</th>
      <th>won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Joe Biden</td>
      <td>DEM</td>
      <td>44552</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Donald Trump</td>
      <td>REP</td>
      <td>41009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Jo Jorgensen</td>
      <td>LIB</td>
      <td>1044</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Howie Hawkins</td>
      <td>GRN</td>
      <td>420</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Delaware</td>
      <td>New Castle County</td>
      <td>Joe Biden</td>
      <td>DEM</td>
      <td>195034</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
president_county_total_votes = president_county_candidate.groupby(['state', 'county'])['total_votes'].sum().to_frame()
```


```python
county = county_population.merge(president_county_total_votes, on=['state', 'county'], how='left')
county.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Population</th>
      <th>county</th>
      <th>state</th>
      <th>total_votes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55869</td>
      <td>Autauga County</td>
      <td>Alabama</td>
      <td>27770.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>223234</td>
      <td>Baldwin County</td>
      <td>Alabama</td>
      <td>109679.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24686</td>
      <td>Barbour County</td>
      <td>Alabama</td>
      <td>10518.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22394</td>
      <td>Bibb County</td>
      <td>Alabama</td>
      <td>9595.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57826</td>
      <td>Blount County</td>
      <td>Alabama</td>
      <td>27588.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
county[county['total_votes'] * 2 > county['Population']].shape[0]
```




    1434



有1434个县满足总投票数超过县人口数的一半

**第2问：计算候选人在各州的总票数**


```python
# 计算候选人在各州的总票数
candidate_votes = president_county_candidate.pivot_table(index = 'state', columns = 'candidate', values = 'total_votes', aggfunc = 'sum', margins=True)
```


```python
# 候选人在全美的总票数排序
candidate_votes = candidate_votes.T.sort_values(['All'], ascending=False).T
```


```python
# 删除边际索引
candidate_votes.drop(index='All', columns='All', inplace=True)
```


```python
# nan填充0
candidate_votes.fillna(value=0, inplace=True)
```


```python
# 删除多余的索引名和列名
candidate_votes.index.name = ""
candidate_votes.columns.name = ""
```


```python
candidate_votes
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Joe Biden</th>
      <th>Donald Trump</th>
      <th>Jo Jorgensen</th>
      <th>Howie Hawkins</th>
      <th>Write-ins</th>
      <th>Rocky De La Fuente</th>
      <th>Gloria La Riva</th>
      <th>Kanye West</th>
      <th>Don Blankenship</th>
      <th>Brock Pierce</th>
      <th>...</th>
      <th>Tom Hoefling</th>
      <th>Ricki Sue King</th>
      <th>Princess Jacob-Fambro</th>
      <th>Blake Huber</th>
      <th>Richard Duncan</th>
      <th>Joseph Kishore</th>
      <th>Jordan Scott</th>
      <th>Gary Swing</th>
      <th>Keith McCormic</th>
      <th>Zachary Scalf</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>Alabama</th>
      <td>849648.0</td>
      <td>1441168.0</td>
      <td>25176.0</td>
      <td>0.0</td>
      <td>7312.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>153405.0</td>
      <td>189892.0</td>
      <td>8896.0</td>
      <td>0.0</td>
      <td>34210.0</td>
      <td>318.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1127.0</td>
      <td>825.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>1672143.0</td>
      <td>1661686.0</td>
      <td>51465.0</td>
      <td>0.0</td>
      <td>2032.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>423932.0</td>
      <td>760647.0</td>
      <td>13133.0</td>
      <td>2980.0</td>
      <td>0.0</td>
      <td>1321.0</td>
      <td>1336.0</td>
      <td>4099.0</td>
      <td>2108.0</td>
      <td>2141.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>California</th>
      <td>11109764.0</td>
      <td>6005961.0</td>
      <td>187885.0</td>
      <td>81025.0</td>
      <td>80.0</td>
      <td>60155.0</td>
      <td>51036.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>1804352.0</td>
      <td>1364607.0</td>
      <td>52460.0</td>
      <td>8986.0</td>
      <td>0.0</td>
      <td>636.0</td>
      <td>1035.0</td>
      <td>8089.0</td>
      <td>5061.0</td>
      <td>572.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>495.0</td>
      <td>355.0</td>
      <td>0.0</td>
      <td>196.0</td>
      <td>175.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Connecticut</th>
      <td>1080680.0</td>
      <td>715291.0</td>
      <td>20227.0</td>
      <td>7538.0</td>
      <td>544.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Delaware</th>
      <td>296268.0</td>
      <td>200603.0</td>
      <td>5000.0</td>
      <td>2139.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>District of Columbia</th>
      <td>317323.0</td>
      <td>18586.0</td>
      <td>2036.0</td>
      <td>1726.0</td>
      <td>3137.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>693.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>5297045.0</td>
      <td>5668731.0</td>
      <td>70324.0</td>
      <td>14721.0</td>
      <td>1055.0</td>
      <td>5966.0</td>
      <td>5712.0</td>
      <td>0.0</td>
      <td>3902.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>2473633.0</td>
      <td>2461854.0</td>
      <td>62229.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Hawaii</th>
      <td>366130.0</td>
      <td>196864.0</td>
      <td>5539.0</td>
      <td>3822.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>931.0</td>
      <td>1183.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Idaho</th>
      <td>287021.0</td>
      <td>554119.0</td>
      <td>16404.0</td>
      <td>0.0</td>
      <td>745.0</td>
      <td>1491.0</td>
      <td>0.0</td>
      <td>3632.0</td>
      <td>1886.0</td>
      <td>2808.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>4253153.0</td>
      <td>2823926.0</td>
      <td>76122.0</td>
      <td>36337.0</td>
      <td>5405.0</td>
      <td>0.0</td>
      <td>9423.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Indiana</th>
      <td>1242495.0</td>
      <td>1729852.0</td>
      <td>58900.0</td>
      <td>0.0</td>
      <td>1951.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>759061.0</td>
      <td>897672.0</td>
      <td>19637.0</td>
      <td>3075.0</td>
      <td>4337.0</td>
      <td>1082.0</td>
      <td>0.0</td>
      <td>3210.0</td>
      <td>1707.0</td>
      <td>544.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kansas</th>
      <td>558669.0</td>
      <td>758100.0</td>
      <td>29797.0</td>
      <td>0.0</td>
      <td>3001.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kentucky</th>
      <td>772474.0</td>
      <td>1326646.0</td>
      <td>26234.0</td>
      <td>0.0</td>
      <td>1332.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6483.0</td>
      <td>0.0</td>
      <td>3599.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Louisiana</th>
      <td>856034.0</td>
      <td>1255776.0</td>
      <td>21645.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>987.0</td>
      <td>4897.0</td>
      <td>860.0</td>
      <td>749.0</td>
      <td>...</td>
      <td>668.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>430473.0</td>
      <td>359899.0</td>
      <td>14005.0</td>
      <td>8111.0</td>
      <td>81.0</td>
      <td>1171.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Maryland</th>
      <td>1985023.0</td>
      <td>976414.0</td>
      <td>33488.0</td>
      <td>15799.0</td>
      <td>20422.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Massachusetts</th>
      <td>2382202.0</td>
      <td>1167202.0</td>
      <td>47013.0</td>
      <td>18658.0</td>
      <td>16327.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Michigan</th>
      <td>2804040.0</td>
      <td>2649852.0</td>
      <td>60381.0</td>
      <td>13718.0</td>
      <td>1090.0</td>
      <td>2986.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7235.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Minnesota</th>
      <td>1717077.0</td>
      <td>1484065.0</td>
      <td>34976.0</td>
      <td>10033.0</td>
      <td>9965.0</td>
      <td>5611.0</td>
      <td>1210.0</td>
      <td>7940.0</td>
      <td>0.0</td>
      <td>5651.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Mississippi</th>
      <td>539398.0</td>
      <td>756764.0</td>
      <td>8026.0</td>
      <td>1498.0</td>
      <td>1423.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3657.0</td>
      <td>1279.0</td>
      <td>659.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>1253014.0</td>
      <td>1718736.0</td>
      <td>41205.0</td>
      <td>8283.0</td>
      <td>805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3919.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Montana</th>
      <td>244786.0</td>
      <td>343602.0</td>
      <td>15252.0</td>
      <td>0.0</td>
      <td>2110.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Nebraska</th>
      <td>374583.0</td>
      <td>556846.0</td>
      <td>20283.0</td>
      <td>0.0</td>
      <td>4667.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Nevada</th>
      <td>703486.0</td>
      <td>669890.0</td>
      <td>14783.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3138.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>New Hampshire</th>
      <td>424921.0</td>
      <td>365654.0</td>
      <td>13235.0</td>
      <td>0.0</td>
      <td>620.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>New Jersey</th>
      <td>2608335.0</td>
      <td>1883274.0</td>
      <td>31677.0</td>
      <td>14202.0</td>
      <td>14881.0</td>
      <td>2728.0</td>
      <td>2928.0</td>
      <td>0.0</td>
      <td>2954.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>New Mexico</th>
      <td>501614.0</td>
      <td>401894.0</td>
      <td>12585.0</td>
      <td>4426.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1640.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>5244006.0</td>
      <td>3250230.0</td>
      <td>60369.0</td>
      <td>32822.0</td>
      <td>3469.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22650.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>2684292.0</td>
      <td>2758773.0</td>
      <td>48678.0</td>
      <td>12194.0</td>
      <td>13315.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7549.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>North Dakota</th>
      <td>114902.0</td>
      <td>235595.0</td>
      <td>9393.0</td>
      <td>0.0</td>
      <td>1929.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>2679165.0</td>
      <td>3154834.0</td>
      <td>67569.0</td>
      <td>18812.0</td>
      <td>1822.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <td>503890.0</td>
      <td>1020280.0</td>
      <td>24731.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5597.0</td>
      <td>0.0</td>
      <td>2547.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>1340383.0</td>
      <td>958448.0</td>
      <td>41582.0</td>
      <td>11831.0</td>
      <td>17089.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Pennsylvania</th>
      <td>3459923.0</td>
      <td>3378263.0</td>
      <td>79397.0</td>
      <td>0.0</td>
      <td>7672.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Rhode Island</th>
      <td>306210.0</td>
      <td>199837.0</td>
      <td>5047.0</td>
      <td>0.0</td>
      <td>2756.0</td>
      <td>923.0</td>
      <td>843.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>South Carolina</th>
      <td>1091541.0</td>
      <td>1385103.0</td>
      <td>27916.0</td>
      <td>6907.0</td>
      <td>0.0</td>
      <td>1862.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>South Dakota</th>
      <td>150471.0</td>
      <td>261043.0</td>
      <td>11095.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <td>1143913.0</td>
      <td>1852948.0</td>
      <td>29883.0</td>
      <td>4545.0</td>
      <td>860.0</td>
      <td>1860.0</td>
      <td>2303.0</td>
      <td>10281.0</td>
      <td>5365.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>5259126.0</td>
      <td>5890347.0</td>
      <td>126243.0</td>
      <td>33396.0</td>
      <td>8799.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>560282.0</td>
      <td>865140.0</td>
      <td>38447.0</td>
      <td>5053.0</td>
      <td>612.0</td>
      <td>0.0</td>
      <td>1139.0</td>
      <td>7213.0</td>
      <td>5551.0</td>
      <td>2623.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Vermont</th>
      <td>242820.0</td>
      <td>112704.0</td>
      <td>3608.0</td>
      <td>1310.0</td>
      <td>1942.0</td>
      <td>48.0</td>
      <td>166.0</td>
      <td>1269.0</td>
      <td>208.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>54.0</td>
      <td>213.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>126.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <td>2413568.0</td>
      <td>1962430.0</td>
      <td>64761.0</td>
      <td>0.0</td>
      <td>19765.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>2369612.0</td>
      <td>1584651.0</td>
      <td>80500.0</td>
      <td>18289.0</td>
      <td>27252.0</td>
      <td>0.0</td>
      <td>4840.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>West Virginia</th>
      <td>235984.0</td>
      <td>545382.0</td>
      <td>10687.0</td>
      <td>2599.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <td>1630673.0</td>
      <td>1610065.0</td>
      <td>38491.0</td>
      <td>0.0</td>
      <td>7721.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5144.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>73491.0</td>
      <td>193559.0</td>
      <td>5768.0</td>
      <td>0.0</td>
      <td>1739.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2208.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
<p>51 rows × 38 columns</p>
</div>



**第3问：找出所有的Biden State**


```python

```
