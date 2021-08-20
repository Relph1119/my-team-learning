# Task03 学习BERT和GPT

## 1 BERT

### 1.1 BERT前景
- 易用性：NLP社区提供强大的组件，下载并方便的使用
- 突破：BERT模型打破了基于语言处理任务的几个记录
- 好处：组件化，节省从零开始训练语言处理模型的时间、精力、知识和资源
- 开发步骤：
  1. 下载预训练模型
  2. 模型微调

### 1.2 句子分类任务
- 微调：需要训练分类器，在训练过程中几乎不用改动BERT模型
- 起源：Semi-supervised Sequence Learning 和 ULMFiT
- 应用场景：垃圾邮件分类器、电影/产品评价分类器

### 1.3 模型架构
- 按模型规模分类：
  1. BERT BASE：与OpenAI的Transformer大小相当
  2. BERT LARGE：非常巨大的模型
- 区别与特点：
  1. BERT BASE的Encoder有12层，LARGE版本有20层
  2. BERT BASE的前馈神经网络有768个隐藏单元，LARGE版本有1024个
  3. BERT BASE的Attention Heads有12个，LARGE版本有16个
- 模型输入：第1个输入的token是CLS（分类），流程和Transformer的Encoder一样
- 模型输出：第1个输出的向量作为后面分类器的输入，使用单层神经网络作为分类器

### 1.4 词嵌入的新时代
- 词嵌入定义：使用一个向量（一组数字）来表示单词，并捕捉单词的语义以及单词之间的关系
- 语境化词嵌入：可以根据单词在句子语境中的含义，赋予不同的词嵌入
- ELMo（语言建模）：训练预测单词序列中的下一个词，获得语言理解能力
- ULM-FiT：使用一个语言模型和一套流程，有效地为各种任务微调这个语言模型
- OpenAI Transformer：由Transformer Decoder堆叠，包括12个Decoder层

### 1.5 BERT：从Decoder到Encoder
- Masked Language Model(MLM语言模型)：使用mask，把需要预测的词屏蔽掉
- 应用场景：
  1. 两个句子任务：预训练阶段给出两个句子的相邻关系、句子分类任务
  2. 特征提取：创建语境化词嵌入
- BERT的使用：
  1. 使用在 Google Colab 上的 BERT FineTuning with Cloud TPUs
  2. 查看仓库代码：
      - 模型定义：modeling.py（class BertModel）
      - 微调网络：run_classifier.py，构建了监督模型分类层
      - 下载预训练模型：BERT Base、BERT Large，以及英语、中文和包括 102 种语言的多语言模型
      - 关注WordPiece：tokenization.py，单词转换工具

## 2 GPT

### 2.1 GPT2和语言模型

- 语言模型
  1. 概念：根据句子的一部分预测下一个词
  2. 应用示例：手机键盘，根据输入的内容，提示下一个单词
  3. GPT2：训练模型通过40GB的WebText数据集训练得到，最小的GPT-2变种需要500MB空间，最大的模型占用超过6.5GB
  4. GPT2的类别：通过不同的Decoder层区分不同的GPT-2

- 与BERT的不同之处
  1. GPT2使用Transformer的Decoder模块构建
  2. 工作方式（自回归）：
      - 产生每个token
      - 将token添加到输入的序列中，形成一个新序列
      - 将上述新序列作为模型下一个时间步的输入

- Transformer模块的进化
  1. Encoder模块：接受特定长度的输入，如果不足，填充序列其余部分
  2. Decoder模块：使用Masked Self-Attention，只允许处理以前和现在的token，可处理多大4000个token

- GPT2简介
  1. 能力：能够处理1024个token，每个token沿着自己的路径经过所有的Decoder模块
  2. 训练方法：生成无条件样本、生成交互式条件样本、生成单词
  3. 训练过程：
      - 并行处理token，生成一个向量
      - 根据模型的词汇表计算一个分数
      - 选择概率最高的词作为输出
      - 把上述输出作为下一次的输入序列，进行下一个预测

- GPT2详解
  1. 查找第一个token \<s\>的embedding向量：在嵌入矩阵中查找输入单词对应的embedding向量
  2. 位置编码：指示单词在序列中的顺序
  3. token流向顺序：
      - 首先通过Self Attention层
      - 通过神经网络层
      - 第一个模块处理token，得到一个结果向量
      - 将结果向量发送到下一个模块
  4. Self-Attention过程：得到输出向量
  5. 模型输出：将上述输出项目乘以嵌入矩阵，得到输出的分数矩阵（单词概率矩阵）
  6. 选取top_k（设置40），即让模型考虑得分最高的40个词
  7. 继续进行上述1~6步迭代，直至生成1024个token，或直到输出表述句子末尾的token

- Self-Attention回顾
  1. 作用：在处理某个词之前，将模型对这个词的相关词和关联词的理解融合起来，通过对句子片段中每个词的相关性打分，并进行标识向量加权求和
  2. 过程：
      - 主要组成部分：Query（当前单词标识，用于对其他所有单词进行评分）、Key（句子中所有单词的标签）、Value（实际的单词表示）
      - 将Query向量与每个单词的Key向量相乘，产生一个分数（点积+Softmax）
      - 每个Value向量乘以上述分数，求和之后得到Self-Attention的输出

### 2.2 可视化Self-Attention

- 原始的Self-Attention
  1. 为每个路径创建 Query、Key、Value 矩阵
  2. 对于每个输入的 token，使用它的 Query 向量为所有其他的 Key 向量进行打分
  3. 将 Value 向量乘以它们对应的分数后求和

- Masked Self-Attention：  
  1. 与上述流程的区别在于第2步，屏蔽当前token之后的词，将这些设置的评分设置为0
  2. 在第2步中，加上一个上三角形的attention mask，将要屏蔽的元素设置为负无穷大
  3. 具体流程：
      - 1) Create q, k, v：将输入乘以第一个权重矩阵，创建Query、Key、Value矩阵
      - 1.5) split attention heads： 将一个长向量变成矩阵
      - 2) Score：进行评分
      - 3) Sum：将每个Value向量乘以对应分数，求和得到输出Z
      - 3.5) Merge attention heads：将输出Z进行合并，连接成一个向量
      - 4) Project：使用第二个权重矩阵进行结果映射
  4. 全连接神经网络
      - 作用：用于处理Self-Attention层的输出
      - 第1层：模型大小的4倍
      - 第2层：把第1层的结果映射回模型的维度

### 2.3 Transformer的Decoder应用

- 机器翻译：不使用Encoder，只用Transformer-Decoder解决翻译问题
- 生成摘要：第一个只是用Decoder来训练的任务，被训练用于阅读一篇维基百科的文章，然后生成摘要
- 迁移学习：在语言模型上进行预训练，然后微调进行生成摘要
- 音乐生成：和语言建模一样，让模型以无监督的方式学习音乐，然后采样输出

## 3 总结

&emsp;&emsp;本次任务，主要介绍了BERT和GPT，其中BERT部分主要围绕BERT前景、句子分类任务的应用、模型架构、词嵌入的扩展介绍，以及BERT的应用场景和使用流程；GPT部分主要通过对语言模型的介绍，展开GPT2的详解，包括GPT2的整个模型训练过程、详细的Self-Attention具体流程介绍（本次是目前关于Self-Attention最为详细的介绍）；最后对Transformer-Decoder应用进行了扩展介绍。
