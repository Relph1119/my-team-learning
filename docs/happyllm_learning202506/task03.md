# Task03 预训练语言模型

## 1 Encoder-only PLM

### 1.1 BERT

- 核心思想：基于Transformer架构，通过将Encoder结构进行堆叠，扩大模型参数；基于双向LSTM架构，在训练数据上预训练，针对下游任务进行微调。
- 模型架构：
    1. 在模型的最顶层加入了一个分类头 prediction_heads（线性层+激活函数），用于将多维度的隐藏状态通过线性层转换到分类维度。
    2. 激活函数是GELU函数：$\displaystyle \text{GELU}(x) = 0.5x(1 + tanh(\sqrt{\frac{2}{\pi}})(x + 0.044715x^3))$
    3. 将相对位置编码融合在了注意力机制中，将相对位置编码同样视为可训练的权重参数。
    4. 在完成注意力分数的计算之后，先通过 Position Embedding 层来融入相对位置信息。

- MLM + NSP（掩码语言模型 + 下一句预测）：
    1. 在MLM训练时，随机选择训练语料中15%的token用于遮蔽，有80%概率被遮蔽，10%被替换为任意一个token，10%保持不变。
    2. NSP主要用于训练模型在句级的语义关系拟合。
    3. 3.3B token的训练语料。

- 下游任务微调：更通用的输入和输出层来适配多任务下的迁移学习。
    1. 文本分类任务：直接修改模型结构中的 prediction_heads 最后的分类头。
    2. 序列标注任务：集成 BERT 多层的隐含层向量再输出最后的标注结果。
    3. 文本生成任务：将Encoder的输出直接解码得到最终生成结果。

### 1.2 RoBERTa

- 模型架构：使用 BERT-large（24层 Encoder Layer、1024的隐藏层维度，总参数量 340M）的模型参数。

- 优化方案：
    1. 去掉 NSP 预训练任务：将 Mask 操作放在训练阶段中，即动态遮蔽策略。
    2. 使用更大规模的预训练数据和预训练步长：使用160GB的预训练数据，训练步长达到500K Step。
    3. 更大的 BEP （字节对编码）词表：使用大小约为 50K 的词表

### 1.3 ALBERT

- 优化方案：
    1. 将 Embedding 参数进行分解：在 Embedding 层的后面加入一个线性矩阵进行维度变换。
    2. 跨层进行参数共享：仅初始化了一个 Encoder 层。
    3. 提出 SOP 预训练任务：正例同样由两个连续句子组成，但负例是将这两个的顺序反过来。

## 2 Encoder-Decoder PLM

### 2.1 T5

- 核心思想：使用基于Tranformer的Encoder和Decoder，使用自注意力机制和多头注意力捕捉全局依赖关系，利用相对位置编码处理长序列中的位置信息，并在每层中包含前馈神经网络进一步处理特征。
- 模型架构：
    1. 编码器用于处理输入文本，解码器用于生成输出文本。
    2. LayerNorm 采用了 RMSNorm：$\displaystyle \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}w_i^2 + \epsilon}}$

- 预训练任务：基于MLM的预训练任务（随机遮蔽15%的token），使用Colossal Clean Crawled Corpus（C4）大规模数据集。

- 大一统思想：所有的 NLP 任务都可以统一为文本到文本的任务。对于不同的NLP任务，每次输入前都会加上一个任务描述前缀，明确指定当前任务的类型。

## 3 Decoder-Only PLM

### 3.1 GPT

- 模型架构：
    1. GPT 使用Decoder-Only 结构，适用于文本生成任务。
    2. GPT 沿用了 Transformer 的经典 Sinusoidal 位置编码。
    3. 掩码自注意力的计算：在计算得到注意力权重之后，通过掩码矩阵来遮蔽了未来 token 的注意力权重，从而限制每一个 token 只能关注到它之前 token 的注意力。
    4. GPT 的 MLP 层没有选择线性矩阵来进行特征提取，而是选择了两个一维卷积核来提取。

- 预训练任务：使用 CLM（因果语言模型），基于一个自然语言序列的前面所有 token 来预测下一个 token。

### 3.2 LLaMA

- 模型架构：
    1. Attention：模型会分别计算query、key和value这三个向量，与GPT不同的就在这个Query和Key的计算。
    2. MLP：通过两个全连接层对hidden_states进行进一步的特征提取。第一个全连接层将hidden_states映射到一个中间维度，然后通过激活函数进行非线性变换。第二个全连接层则将特征再次映射回原始的hidden_states维度。

### 3.3 GLM

- 模型架构：
    1. 使用 Post Norm 而非 Pre Norm。
    2. 使用单个线性层实现最终 token 的预测，而不是使用 MLP。
    3. 激活函数从 ReLU 换成了 GeLUS。

- 预训练任务：
    1. 结合自编码思想和自回归思想的预训练方法。
    2. 优化自回归空白填充任务来实现 MLM 与 CLM 思想的结合，每次遮蔽一连串 token。
