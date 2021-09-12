# 我的组队学习
&emsp;&emsp;记录我参加的Datawhale组队学习，包括笔记和习题解答。

## 在线阅读地址
在线阅读地址：https://relph1119.github.io/my-team-learning

## 项目结构
<pre>
docs---------------------------------------学习笔记
notebook-----------------------------------JupyterNotebook格式笔记
+---pandas20-------------------------------第20期Pandas学习
|   +---asserts----------------------------joyful-pandas教材
|   +---data-------------------------------数据集
|   +---my_homework------------------------我的笔记
|   +---source-----------------------------教材中的图片资源
+---knowledge_graph_basic21----------------第21期自然语言处理实践（知识图谱）
|   +---asserts----------------------------知识图谱组队学习教材
|   +---my_homework------------------------我的笔记
+---ensemble_learning23--------------------第23期集成学习
|   +---asserts----------------------------集成学习组队学习教材
|   +---my_homework------------------------我的笔记  
+---ensemble_learning24--------------------第24期集成学习
|   +---asserts----------------------------集成学习组队学习教材
|   +---my_homework------------------------我的笔记
+---ensemble_learning25--------------------第25期集成学习
|   +---asserts----------------------------集成学习组队学习教材
|   +---my_homework------------------------我的笔记  
+---gnn_learning26-------------------------第26期图神经网络  
|   +---asserts----------------------------图神经网络组队学习教材  
|   +---my_homework------------------------我的笔记  
+---pumpkin_learning27---------------------第27期吃瓜课程（西瓜书+南瓜书）
+---transformers_nlp28---------------------第28期基于Transformers的自然语言处理
QASystemOnMedicalGraph---------------------基于医疗领域知识图谱的问答系统源码
requirements.txt---------------------------运行环境依赖包
</pre>

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
