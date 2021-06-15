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

## 环境安装
### Python版本
Python 3.7.9

### Notebook运行环境配置
安装相关的依赖包
```shell
pip install -r requirements.txt
```

### pytorch安装
执行以下命令安装pytorch
```shell
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### pytorch geometric安装
执行以下命令安装pytorch geometric
```shell
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

### Neo4j安装
- [Windows10下安装Neo4j参考文档](https://blog.csdn.net/lihuaqinqwe/article/details/80314895)  
- 如果是JDK1.8，可下载[Neo4j V3.5.26版本](https://go.neo4j.com/download-thanks.html?edition=community&release=3.5.26&flavour=winzip&_gl=1*cfbj98*_ga*MjIzOTA4ODkzLjE2MTAyOTEzODU.*_ga_DL38Q8KGQC*MTYxMDI5MTM4NS4xLjEuMTYxMDI5NDI0NS4w&_ga=2.141402866.1342715293.1610291386-223908893.1610291385)
