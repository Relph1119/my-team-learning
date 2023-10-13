# 我的组队学习
记录我参加的Datawhale组队学习，包括学习笔记和习题解答。

## 在线阅读地址
在线阅读地址：https://relph1119.github.io/my-team-learning

## 项目结构
<pre>
docs---------------------------------------学习笔记
notebook-----------------------------------JupyterNotebook格式笔记
+---pandas20---------------------------------第20期Pandas学习
+---knowledge_graph_basic21------------------第21期自然语言处理实践（知识图谱）
+---ensemble_learning23----------------------第23期集成学习
+---ensemble_learning24----------------------第24期集成学习
+---ensemble_learning25----------------------第25期集成学习
+---gnn_learning26---------------------------第26期图神经网络  
+---pumpkin_learning27-----------------------第27期吃瓜课程（西瓜书+南瓜书）
+---transformers_nlp28-----------------------第28期基于Transformers的自然语言处理
+---matplotlib_learning29--------------------第29期数据可视化
+---tree_ensemble30--------------------------第30期树模型与集成学习
+---unusual_deep_learning31------------------第31期水很深的深度学习
+---recommender_system32---------------------第32期推荐系统
+---pytorch_learning35-----------------------第35期深入浅出Pytorch
+---lee_ml37---------------------------------第37期李宏毅机器学习
+---pytorch_rechub_learning38----------------第38期使用PyTorch复现推荐模型
+---intel_openvino_learning39----------------第39期Intel带你初识视觉识别
+---intel_openvino_advanced_learning40-------第40期Intel OpenVINO高级课程
+---Interpretable_machine_learning44---------第44期可解释性机器学习
+---cs224w_learning46------------------------第46期CS224W图机器学习
+---diffusion_model_learning51---------------第51期扩散模型从原理到实战
QASystemOnMedicalGraph---------------------基于医疗领域知识图谱的问答系统源码
requirements.txt---------------------------运行环境依赖包
</pre>

## 学习资料
【1】[第20期组队学习-Pandas（joyful-pandas）](https://datawhalechina.github.io/joyful-pandas)  
【2】[第21期组队学习-自然语言处理实践（知识图谱）](https://github.com/datawhalechina/team-learning-nlp/tree/master/KnowledgeGraph_Basic)  
【3】[基于医疗领域知识图谱的问答系统](https://github.com/zhihao-chen/QASystemOnMedicalGraph)  
【4】[第23/24/25期组队学习-集成学习](https://github.com/datawhalechina/team-learning-data-mining/tree/master/EnsembleLearning)  
【5】[第26期组队学习-图神经网络](https://github.com/datawhalechina/team-learning-nlp/tree/master/GNN)  
【6】[第27期组队学习-吃瓜教程](https://www.bilibili.com/video/BV1Mh411e7VU)  
【7】[第28期组队学习-基于Transformers的自然语言处理](https://github.com/datawhalechina/learn-nlp-with-transformers)  
【8】[第29期组队学习-Matplotlib可视化](https://github.com/datawhalechina/fantastic-matplotlib)    
【9】[Matplotlib 50题从入门到精通](https://www.heywhale.com/mw/notebook/5ec2336f693a730037a4415c)  
【10】[第30期组队学习-树模型与集成学习](https://datawhalechina.github.io/machine-learning-toy-code/)  
【11】[第31期组队学习-水很深的深度学习](https://datawhalechina.github.io/unusual-deep-learning)  
【12】[第32期组队学习-推荐系统](https://github.com/datawhalechina/fun-rec)  
【13】[第35期组队学习-深入浅出Pytorch](https://github.com/datawhalechina/thorough-pytorch)  
【14】[第37期组队学习-李宏毅机器学习](https://github.com/datawhalechina/leeml-notes)  
【15】[第38期组队学习-使用PyTorch复现推荐模型](https://www.wolai.com/rechub/2qjdg3DPy1179e1vpcHZQC)  
【16】[OpenVINO for CV Applications(Beginner Level)](https://vxr.h5.xeknow.com/s/3Eg4J8)  
【17】[OV300 for CV Applications Advanced Level](https://vxr.h5.xeknow.com/s/204VNE)  
【18】[可解释性机器学习-同济子豪兄](https://datawhaler.feishu.cn/docx/OTROd2zCIoZlLyxjhSKclxKNnDe)  
【19】[CS224W图机器学习-同济子豪兄](https://github.com/TommyZihao/zihao_course/tree/main/CS224W)  
【20】[Hugging Face Diffusion Models](https://github.com/huggingface/diffusion-models-class)  

## 环境安装
### Python版本
Python 3.8 Windows环境

### 运行环境配置
安装相关的依赖包
```shell
pip install -r requirements.txt
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

### 本地启动docsify
```shell
docsify serve ./docs
```