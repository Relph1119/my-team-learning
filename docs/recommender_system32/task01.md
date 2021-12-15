# Task01 熟悉新闻推荐系统基本流程

![新闻推荐系统功能图](images/news-rec-sys-function-diagram.png)

&emsp;&emsp;根据新闻推荐系统功能图，主要分为`Online`（在线）和`Offline`（离线，也称为线下）两个的部分。

## 1 功能介绍
### 1.1 `Offline`离线部分

#### 1.1.1 物料爬取与处理
- 物料爬取：主要采用scrapy爬虫工具，在每天晚上23点将当天的新闻内容从网页中进行抓取，存入MongoDB的SinaNews数据库中，具体执行代码：`materials/news_scrapy/sinanews/run.py`
- 当天新闻动态画像更新：将用户对前一天新闻的交互，包括阅读、点赞和收藏等行为（动态）、存入Redis中，具体执行代码（`materials/process_material.py`）：
```python
# 画像处理
protrail_server = NewsProtraitServer()
```

- 物料画像处理：将当天新闻内容通过特征画像提取从MongoDB存入Redis中，具体执行代码(`materials/process_material.py`)：
```python
# 处理最新爬取新闻的画像，存入特征库
protrail_server.update_new_items()
# 更新新闻动态画像, 需要在redis数据库内容清空之前执行
protrail_server.update_dynamic_feature_protrail()
# 生成前端展示的新闻画像，并在mongodb中备份一份
protrail_server.update_redis_mongo_protrail_data()
```

- 前端新闻展示：主要将Redis中的新闻动态画像数据提供给前端进行展示，具体执行代码：`materials/update_redis.py`

#### 1.1.2 用户画像更新
- 用户注册：用户通过前端注册页面，进行用户注册，将用户信息存入MySQL的用户注册信息表（register_user）中

- 用户行为：用户通过阅读、点赞及收藏新闻，将用户行为数据存入MySQL的用户阅读信息表（user_read）、用户点赞信息表（user_likes）和用户收藏信息表（user_collections）

- 用户画像更新：将当天的新注册用户基本信息及其行为数据构造用户画像，存入MongoDB中的`UserProtrail`，具体执行代码：`materials/process_user.py`

#### 1.1.3 热门页列表及推荐页列表展示
- 热门页列表数据生成：通过物料画像处理，根据特征库（MongoDB中的`FeatureProtrail`）的最新特征计算新闻热度，过滤10天前的新闻，生成热门列表模板，并更新每个用户的热门页列表，采用倒排索引方式，将数据存入Redis中，具体执行代码（`recprocess/offline.py`）:
```python
    def hot_list_template_to_redis(self):
        """热门页面，初始化的时候每个用户都是同样的内容
        """
        self.hot_recall.update_hot_value()
        # 将新闻的热度模板添加到redis中
        self.hot_recall.group_cate_for_news_list_to_redis()
        print("a hot rec list are saved into redis.....")
```

- 推荐页列表数据生成：根据用户画像（MongoDB中的`UserProtrail`）将用户分为4类人群，分别生成冷启动的新闻模板，具体执行代码（`recprocess/offline.py`）:
```python
    def cold_start_template_to_redis(self):
        """冷启动列表模板
        """
        # 生成冷启动模板
        self.cold_start.generate_cold_user_strategy_templete_to_redis_v2()
        # 初始化已有用户的冷启动列表
        self.cold_start.user_news_info_to_redis()
```

### 1.2 `Online`在线部分

#### 1.2.1 获取推荐页列表

&emsp;&emsp;通过用户id，判断该用户是新老用户，对于新用户，根据当前新闻热度信息，给用户生成冷启动的推荐列表，并存入Redis中，具体执行代码（`server.py`的`rec_list()`函数）：

```python
rec_news_list = recsys_server.get_cold_start_rec_list_v2(user_id, age, gender)
```

&emsp;&emsp;对于老用户，进行个性化推荐，通过召回→排序→重排等，给老用户生成一份个性化列表，并存入Redis中

#### 1.2.2 获取热门页列表

&emsp;&emsp;通过用户id，判断该用户是否已存在热门页列表数据，如果不存在则是新用户，就从热门列表模板中拷贝一份，如果存在则是老用户，直接获取热门列表数据即可；之后过滤当前曝光列表，排序列表并更新至Redis中，具体执行代码（`server.py`的`hot_list()`函数）：

```python
rec_news_list = recsys_server.get_hot_list_v2(user_id)
```

#### 1.2.3 获取新闻详情页

&emsp;&emsp;主要展示新闻详细信息，并提供用户点赞和收藏功能，具体执行代码：`server.py`的`news_detail()`函数

## 2 数据流向

### 2.1 `Offline`数据流向
- 物料数据处理的数据流向：
  1. 系统将当天新闻数据爬取存入MongoDB中
  2. 系统更新前一天新闻动态画像数据（用户对新闻的行为数据）存入Redis中
  3. 以上两个数据通过物料画像处理程序，将特征画像更新到MongoDB中
  4. 生成热门页列表数据和推荐页列表数据，并存入Redis中

- 用户画像更新的数据流向：
  1. 用户注册之后，系统将用户注册信息存入MySQL中
  2. 用户对新闻进行阅读、点赞及收藏操作后，系统将行为数据存入MySQL中
  3. 将用户注册信息和其行为数据构建用户画像存入MongoDB中 

### 2.2 `Online`数据流向
- 生成推荐页列表的数据流向：
  1. 根据用户ID，从Redis中获取推荐页列表数据
  2. 从Redis中获取用户曝光表，并对推荐页列表数据进行过滤
  3. 更新用户曝光表，并存入Redis中
  4. 生成详情信息，并存入Redis中
  5. 生成排序列表，再前端页面上展示

- 生成热门页列表的数据流向：
  1. 根据用户ID，从Redis中获取热门页列表数据
  2. 从Redis中获取用户曝光表，并对热门页列表数据进行过滤
  3. 更新用户曝光表，并存入Redis中
  4. 生成详情信息，并存入Redis中
  5. 生成排序列表，再前端页面上展示

## 3 交互请求
后端请求（具体代码：`server.py`）：  
- 用户注册请求：`register()`函数
- 用户登录请求：`login()`函数
- 用户推荐页请求：`rec_list()`函数
- 用户热门页请求：`hot_list()`函数
- 新闻详情页请求：`news_detail()`函数
- 用户行为请求（阅读、点赞和收藏）：`actions()`函数

## 4 总结
&emsp;&emsp;本次任务，主要熟悉新闻推荐系统基本流程，包括`Offline`和`Online`部分：
1. 主要功能：`Offline`的主要功能有新闻数据爬取与处理、用户画像更新、热门列表及推荐页列表展示；`Online`的主要功能有获取推荐页列表、获取热门页列表、获取新闻详情页；
2. 数据流向：`Offline`包括物料数据处理、用户画像更新的数据流向；`Online`包括生成推荐页列表、生成热门页列表的数据流向
3. 交互请求：主要包括用户注册请求、用户登录请求、用户推荐页请求、用户热门页请求、新闻详情页请求、用户行为请求