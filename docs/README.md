# 我的组队学习 {docsify-ignore-all}
记录我参加的Datawhale组队学习，包括笔记和习题解答。

## 在线阅读地址
在线阅读地址：https://relph1119.github.io/my-team-learning

## 目录
- 第29期数据可视化
    - Task01 Matplotlib初相识
    - Task02 艺术画笔见乾坤
    - Task03 布局格式定方圆
    - Task04 文字图例尽眉目
    - Task05 样式色彩秀芳华
- 第28期基于Transformers的自然语言处理
    - Task01 学习概览
    - Task02 学习Attention和Transformer
    - Task03 学习BERT和GPT
    - Task04 编写BERT模型
    - Task05 BERT应用到下游任务、训练和优化
    - Task06 Transformers解决文本分类任务、超参搜索
    - Task07 Transformers解析序列标注任务
    - Task08 Transformers解决抽取式问答任务
    - Task09 Transformers解决机器翻译任务
- 第27期吃瓜教程
    - Task01 概览西瓜书+南瓜书第1、2章
    - Task02 详读西瓜书+南瓜书第3章
    - Task03 详读西瓜书+南瓜书第4章
    - Task04 详读西瓜书+南瓜书第5章
    - Task05 详读西瓜书+南瓜书第6章
- 第26期图神经网络
    - Task01 简单图论、环境配置与PyG库
    - Task02 消息传递范式
    - Task03 基于图神经网络的节点表征学习
    - Task04 数据完整存储与内存的数据集类+节点预测与边预测任务实践
    - Task05 超大图上的节点表征学习
    - Task06 基于图神经网络的图表示学习
    - Task07 图预测任务实践
    - Task08 总结
- 第25期集成学习
    - Task12 Blending集成学习算法
    - Task13 Stacking集成学习算法
    - Task14 集成学习案例一（幸福感预测）
    - Task15 集成学习案例二（蒸汽量预测）
- 第24期集成学习
    - Task07 投票法的原理和案例分析
    - Task08 Bagging的原理和案例分析
    - Task09 Boosting的思路与Adaboost算法
    - Task10 前向分步算法与梯度提升决策树
    - Task11 XGBoost算法分析与案例调参实例
- 第23期集成学习
    - Task01 熟悉机器学习的三大主要任务
    - Task02 掌握基本的回归模型
    - Task03 掌握偏差与方差理论
    - Task04 掌握回归模型的评估及超参数调优
    - Task05 掌握基本的分类模型
    - Task06 掌握分类问题的评估及超参数调优
- 第21期自然语言处理实践（知识图谱）
    - Task01 知识图谱介绍
    - Task02 基于医疗知识图谱的问答系统操作介绍
    - Task03 Neo4j图数据库导入数据
    - Task04 用户输入->知识库的查询语句
    - Task05 Neo4j 图数据库查询
- 第20期Pandas组队学习
    - Task01 预备知识
    - Task02 Pandas基础
    - Task03 索引
    - Task04 分组
    - Task05 变形
    - Task06 连接
    - Task Special
    - Task07 缺失数据
    - Task08 文本数据
    - Task09 分类数据
    - Task10 时序数据
    - Task11 综合练习

## 学习资料
【1】[第20期组队学习-Pandas](http://datawhale.club/t/topic/580)  
【2】[joyful-pandas](https://datawhalechina.github.io/joyful-pandas/build/html/%E7%9B%AE%E5%BD%95/index.html)  
【3】[第21期组队学习-自然语言处理实践（知识图谱）](http://datawhale.club/t/topic/1010)   
【4】[KnowledgeGraph_Basic](https://github.com/datawhalechina/team-learning-nlp/tree/master/KnowledgeGraph_Basic)  
【5】[基于医疗领域知识图谱的问答系统](https://github.com/zhihao-chen/QASystemOnMedicalGraph)  
【6】[第23/24/25期组队学习-集成学习](https://github.com/datawhalechina/team-learning-data-mining/tree/master/EnsembleLearning)  
【7】[第26期组队学习-图神经网络](https://github.com/datawhalechina/team-learning-nlp/tree/master/GNN)    
【8】[第27期组队学习-吃瓜教程](https://www.bilibili.com/video/BV1Mh411e7VU)  
【9】[第28期组队学习-基于Transformers的自然语言处理](https://github.com/datawhalechina/learn-nlp-with-transformers)  
【10】[第29期组队学习-Matplotlib可视化](https://github.com/datawhalechina/fantastic-matplotlib)  
【11】[Matplotlib 50题从入门到精通](https://www.heywhale.com/mw/notebook/5ec2336f693a730037a4415c)  

## 环境安装
### Python版本
Mini-Conda Python 3.8 Windows环境

### Notebook运行环境配置
安装相关的依赖包
```shell
conda install --yes --file requirements.txt
```

### 设置Jupyter Notebook代理
```shell
set HTTPS_PROXY=http://127.0.0.1:19180
set HTTP_PROXY=http://127.0.0.1:19180
```
设置代理之后，启动Jupyter Notebook
```shell
jupyter notebook
```

### Neo4j安装
- [Windows10下安装Neo4j参考文档](https://blog.csdn.net/lihuaqinqwe/article/details/80314895)  
- 如果是JDK1.8，可下载[Neo4j V3.5.26版本](https://go.neo4j.com/download-thanks.html?edition=community&release=3.5.26&flavour=winzip&_gl=1*cfbj98*_ga*MjIzOTA4ODkzLjE2MTAyOTEzODU.*_ga_DL38Q8KGQC*MTYxMDI5MTM4NS4xLjEuMTYxMDI5NDI0NS4w&_ga=2.141402866.1342715293.1610291386-223908893.1610291385)

### pytorch安装
执行以下命令安装pytorch
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

### pytorch geometric安装
执行以下命令安装pytorch geometric
```shell
conda install pytorch-geometric -c rusty1s -c conda-forge
```

### ray\[tune\]安装
```shell
conda install ray-tune -c conda-forge
```

### Conda批量导出环境中所有组件
```shell
conda list -e > requirements.txt
```

## 关注我们
<div align=center><img src="res/qrcode.jpeg" width="250" height="270" alt="Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。"></div>