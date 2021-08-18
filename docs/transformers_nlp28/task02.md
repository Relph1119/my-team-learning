# Task02 学习Attention和Transformer

## 1 知识梳理

### 1.1 seq2seq模型

- 定义：  
&emsp;&emsp;一个序列到一个序列（seq2seq）模型，接收的输入是一个（单词、字母、图像特征）序列，输出的是另一个序列；在神经机器翻译中，一个序列是指一连串的单词。


- 模型结构：由编码器和解码器组成
  1. 编码器：用于处理输入序列中的每个元素，把这些信息转换为一个上下文向量，处理完成后，输出到解码器；
  2. 解码器：用于逐项生成输出序列中的元素。
  
  
- 原始模型：
  1. 编码器和解码器使用循环神经网络（RNN）；
  2. 上下文向量的长度基于编码器RNN的隐藏层神经元的数量；
  3. RNN在每个时间步接受2个输入：输入序列中的一个元素、一个隐藏层状态(hidden state)。
  
  
- 处理过程：  
&emsp;&emsp;编解码器处理过程：编码器和解码器在每个时间步处理输入得到输出，RNN根据当前时间步的输入和前一个时间步的隐藏层状态，更新当前时间步的隐藏层状态。

### 1.2 Attention
- 效果：  
&emsp;&emsp;在机器翻译场景，产生英语翻译之前，提供从输入序列放大相关信号的能力，使得模型产生更好的结果。


- Attention与seq2seq模型的区别：
  1. Attention中，编码器把所有时间步的隐藏层状态传递给解码器；而seq2seq中，只将最后一个隐藏层传递给解码器；
  2. Attention的解码器在产生输出序列之前，做了一些处理：
      - 查看所有接收到编码器的hidden state
      - 给每个hidden state一个分数
      - 将每个hidden state乘以经过softmax的对应分数，高分对应的hidden state被放大，低分的则被缩小
  
  
- Attention处理过程：
  1. 解码器RNN的输入：一个embedding向量、一个初始化的解码器hidden state
  2. 处理上述输入，产生一个输出和一个新的hidden state（h4向量）
  3. 使用编码器的hidden state和上述产生的hidden state（h4向量）计算当前时间步的上下文向量（C4向量）
  4. 将h4和C4进行拼接，得到一个向量
  5. 将上述向量输入到一个前馈神经网络
  6. 前馈神经网络的输出表示这个时间步输出的单词
  7. 下一个时间步重复上述步骤（从第1步到第6步）

### 1.3 Transformer
- 主要组成部分：
  1. 由编码部分和解码部分组成
  2. 编码部分由多层编码器组成（6层），解码部分由多层解码器组成（6层）
  3. 编码器由Self-Attention Layer 和 Feed Forward Neural Network（前馈神经网络，FFNN）组成
  4. 解码器由Self-Attention Layer、Encoder-Decoder Attention、FFNN组成，其中Encoder-Decoder Attention用于帮助解码器聚焦于输入句子的相关部分

- Transformer输入  
&emsp;&emsp;使用词嵌入算法，将每个词转换为一个词向量，向量列表的长度为训练集中的句子最大长度。

- Encoder（编码器）  
输入：一个向量列表，上一个编码器的输出  
输出：同样大小的向量列表，连接到下一个编码器  
数据流：
  1. 每个单词转换成一个向量后，输入到self-attention层
  2. 每个位置的单词得到新向量，输入到FFNN

- Self-Attention  
&emsp;&emsp;直观理解，Self-Attention机制使模型不仅能够关注这个位置的词，而且能够关注句子中其他位置的词，作为辅助线索，更好地编码当前位置的词。

- 残差连接：  
&emsp;&emsp;在编解码器的每个子层，都有一个残差连接、一个层标准化（layer-normalization）

- Decoder（解码器）  
  1. 解码阶段的每一个时间步都输出一个翻译后的单词
  2. 将解码器的输入向量，加上位置编码向量，用于指示每个词的位置
  3. Self-Attention层：只允许关注到输出序列中早于当前位置之前的单词，在Self-Attention分数经过Softmax层之前，屏蔽当前位置之后的位置
  4. Encoder-Decoder Attention层：和多头注意力机制类似，但是使用前一层的输出构造Query矩阵，而Key矩阵和Value矩阵来自于解码器最终的输出。

- 最后的线性层和Softmax层
  1. 线性层：把解码器输出的向量，映射到一个更长的向量(logist向量)，和模型的输出词汇表长度一致
  2. Softmax层：把logist向量中的数，转换为概率，取概率最高的那个数字对应的词，就是输出的单词

### 1.4 Self-Attention

- Self-Attention输出的计算步骤：
  1. 对输入编码器的每个词向量，通过乘以三个参数矩阵，创建3个向量（Query向量、Key向量、Value向量）
  2. 计算Attention Score分数：计算某个词的对句子中其他位置的词放置的注意力程度，通过该词对应的Query向量与每个词的Key向量点积计算得到。
  3. 将每个分数除以$\sqrt{d_{key}}$，$d_{key}$是Key向量的长度
  4. 这些分数经过一个Softmax，进行归一化
  5. 将归一化的分数乘以每个Value向量
  6. 对上步的值进行相加，得到该词在Self-Attention层的输出

- Self-Attention的矩阵形式计算  
假设所有词向量组成矩阵$X$，权重矩阵$W^Q,W^K,W^V$
  1. 根据第1步，可得
$$
Q = X \cdot W^Q \\
K = X \cdot W^K \\
V = X \cdot W^V
$$
  2. 根据第2~6步，可得Self-Attention的输出
$$
Z = \text{softmax}\left( \frac{Q \cdot K^T}{\sqrt{d_{K}}} \right) \cdot V
$$

### 1.5 多头注意力机制（multi-head attention）
- 作用：增强Attention层
  1. 扩展了模型关注不同位置的能力
  2. 赋予Attention层多个“子表示空间”


- 处理过程：
  1. 随机初始化8组$W^Q,W^K,W^V$
  2. 根据矩阵计算的第1步，词向量矩阵$X$和每组$W^Q,W^K,W^V$相乘，得到8组$Q,K,V$矩阵
  3. 根据矩阵计算的第2步，得到8个$Z$矩阵
  4. 将8个$Z$矩阵横向拼接，并乘以另一个权重矩阵$W^O$，得到一个矩阵$Z$

### 1.6 使用位置编码表示序列中单词的顺序
- 目的：通过对每个输入的向量，添加一个遵循特定模式向量，用于确定每个单词的我i之，或者句子中不同单词之间的距离。
- 特定模式向量：向量的左半部分值由sine函数产生，右半部分的值由cosine函数产生，然后拼接，得到每个位置编码向量。
- 优点：扩展到需要翻译句子的序列长度

### 1.7 损失函数
- 目的：由于模型的参数是随机初始化的，每个词输出的概率分布与正确的输出概率分布比较，使用反向传播调整模型权重
- 贪婪解码：由于模型每个时间步只产生一个输出，模型是从概率分布中选择概率最大的词，并丢弃其他词
- 集束搜索：每个时间步保留beam_size个概率最高的输出词，然后在下一个时间步，根据第1个词计算第2个位置的词的概率分布，对于后续位置，重复上述过程。
  - beam_size：用于在所有时间步保留最高频率的词的个数
  - top_beams：用于表示最终返回翻译结果的个数

## 2 实战练习

### 2.1 使用PyTorch的MultiheadAttention来实现Attention的计算

torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)

参数说明如下：
  - embed_dim：最终输出的 K、Q、V 矩阵的维度，这个维度需要和词向量的维度一样
  - num_heads：设置多头注意力的数量。如果设置为 1，那么只使用一组注意力。如果设置为其他数值，那么 num_heads 的值需要能够被 embed_dim 整除
  - dropout：这个 dropout 加在 attention score 后面

forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)

参数说明如下：
  - query：对应于 Key 矩阵，形状是 (L,N,E) 。其中 L 是输出序列长度，N 是 batch size，E 是词向量的维度
  - key：对应于 Key 矩阵，形状是 (S,N,E) 。其中 S 是输入序列长度，N 是 batch size，E 是词向量的维度
  - value：对应于 Value 矩阵，形状是 (S,N,E) 。其中 S 是输入序列长度，N 是 batch size，E 是词向量的维度
  - key_padding_mask：如果提供了这个参数，那么计算 attention score 时，忽略 Key 矩阵中某些 padding 元素，不参与计算 attention。形状是 (N,S)。其中 N 是 batch size，S 是输入序列长度。
    - 如果 key_padding_mask 是 ByteTensor，那么非 0 元素对应的位置会被忽略
    - 如果 key_padding_mask 是 BoolTensor，那么  True 对应的位置会被忽略
  - attn_mask：计算输出时，忽略某些位置。形状可以是 2D  (L,S)，或者 3D (N∗numheads,L,S)。其中 L 是输出序列长度，S 是输入序列长度，N 是 batch size。
    - 如果 attn_mask 是 ByteTensor，那么非 0 元素对应的位置会被忽略
    - 如果 attn_mask 是 BoolTensor，那么  True 对应的位置会被忽略 


```python
import torch
from torch import nn

# nn.MultiheadAttention 输入第0维为length
# batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
query = torch.rand(12, 64, 300)
# batch_size 为 64，有 10 个词，每个词的 Key 向量是 300 维
key = torch.rand(10, 64, 300)
# batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
value = torch.rand(10, 64, 300)

embed_dim = 300
num_heads = 10
# 输出是 (attn_output, attn_output_weights)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn.forward(query, key, value)
# output: torch.Size([12, 64, 300])
# batch_size 为 64，有 12 个词，每个词的向量是 300 维
print(attn_output.shape)
```

    torch.Size([12, 64, 300])
    

### 2.2 使用自编程实现Attention的计算


```python
import torch
from torch import nn


class MultiheadAttention(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        # 每个词输出的向量维度
        self.hid_dim = hid_dim
        # 多头注意力的数量
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 n_heads
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # Z: [64,6,12,50]
        Z = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # Z: [64,6,12,50] 转置-> [64,12,6,50]
        Z = Z.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # Z: [64,12,6,50] -> [64,12,300]
        Z = Z.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        Z = self.fc(Z)
        return Z
```


```python
# batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
query = torch.rand(64, 12, 300)
# batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
key = torch.rand(64, 10, 300)
# batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
value = torch.rand(64, 10, 300)
attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
output = attention(query, key, value)
# output: torch.Size([64, 12, 300])
print(output.shape)
```

    torch.Size([64, 12, 300])
    

### 2.3 Transformer代码详解

![transformer-arc](images/task02/transformer-arc.png)

#### 2.3.1 词嵌入

1. 序列预处理：进行词切分成列表，再进行索引转换，最后将文本转换成数组，batch_size个句子，每个句子长度是seq_length
2. 词嵌入处理：将每一个词用预先训练好的向量进行映射，使用torch.nn.Embedding实现


```python
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from typing import List, Optional, Tuple
import math
import warnings
```


```python
# 文本中有2个句子，每个句子4个词
X = torch.zeros((2,4),dtype=torch.long)
# 词表中有10个词，每个词的维度是8
embed = nn.Embedding(10,8)
# 词嵌入处理之后的维度
print(embed(X).shape)
```

    torch.Size([2, 4, 8])
    

#### 2.3.2 位置编码

作用：用于区分不同词，以及同词不同特征之间的关系  
代码思路：
1. 设置Dropout层
2. 初始化矩阵P、X_
3. 构造特定模式的矩阵P，将sine函数生成和cosine函数生成的向量交织在一起
4. 将输入矩阵X进行位置编码
5. 对输入矩阵X进行Dropout


```python
Tensor = torch.Tensor


def positional_encoding(X, num_features, dropout_p=0.1, max_len=512) -> Tensor:
    """
        给输入加入位置编码
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零时执行dropout
        - max_len: 句子的最大长度，默认512

    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]
    """

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)
    return dropout(X)
```


```python
# 位置编码例子
X = torch.randn((2,4,10))
X = positional_encoding(X, 10)
print(X.shape)
```

    torch.Size([2, 4, 10])
    

#### 2.3.3 多头注意力机制

代码思路：
1. 参数初始化
2. multi_head_attention_forward
    - query, key, value通过_in_projection_packed变换得到q,k,v
    - mask机制
    - 点积attention


```python
def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
) -> List[Tensor]:
    """
    用一个大的权重参数矩阵进行线性变换

    参数:
        q, k, v: 对self-Attention来说，三者都是src；对于seq2seq模型，k和v是一致的tensor。
                 但它们的最后一维(num_features或者叫做embed_dim)都必须保持一致。
        w: 用以线性变换的大矩阵，按照q,k,v的顺序压在一个tensor里面。
        b: 用以线性变换的偏置，按照q,k,v的顺序压在一个tensor里面。

    形状:
        输入:
        - q: shape:`(..., E)`，E是词嵌入的维度（下面出现的E均为此意）。
        - k: shape:`(..., E)`
        - v: shape:`(..., E)`
        - w: shape:`(E * 3, E)`
        - b: shape:`E * 3`

        输出:
        - 输出列表 :`[q', k', v']`，q,k,v经过线性变换前后的形状都一致。
    """
    # 得到q的最后一个维度
    E = q.size()[-1]
    # 若为自注意，则q = k = v = src，因此它们的引用变量都是src
    # 即k is v和q is k结果均为True
    # 若为seq2seq，k = v，因而k is v的结果是True
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # seq2seq模型
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)
```


```python
def _scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    在query, key, value上计算点积注意力，若有注意力遮盖则使用，并且应用一个概率为dropout_p的dropout

    参数：
        - q: shape:`(B, Nt, E)` B代表batch size， Nt是目标语言序列长度，E是嵌入后的特征维度
        - key: shape:`(B, Ns, E)` Ns是源语言序列长度
        - value: shape:`(B, Ns, E)`与key形状一样
        - attn_mask: 要么是3D的tensor，形状为:`(B, Nt, Ns)`或者2D的tensor，形状如:`(Nt, Ns)`

        - Output: attention values: shape:`(B, Nt, E)`，与q的形状一致;attention weights: shape:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
        # attn意味着目标序列的每个词对源语言序列做注意力
    attn = F.softmax(attn, dim=-1)
    if dropout_p:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn
```


```python
def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    形状：
        输入：
        - query：`(L, N, E)`
        - key: `(S, N, E)`
        - value: `(S, N, E)`
        - key_padding_mask: `(N, S)`
        - attn_mask: `(L, S)` or `(N * num_heads, L, S)`
        输出：
        - attn_output:`(L, N, E)`
        - attn_output_weights:`(N, L, S)`
    """
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    # query ,key, value变换得到q, k, v
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"

        # 对不同维度的形状判定
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # reshape q,k,v将Batch放在第一维以适合点积注意力
    # 同时为多头机制，将不同的头拼在一起组成一层
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            # 若attn_mask为空，直接用key_padding_mask
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
    # 若attn_mask值是布尔值，则将mask转换为float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # 若training为True时才应用dropout
    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
```


```python
class MultiheadAttention(nn.Module):
    """
    参数：
        embed_dim: 词嵌入的维度
        num_heads: 平行头的数量
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 kdim=None, vdim=None, batch_first=False) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            # 初始化前后形状维持不变
            # (seq_length x embed_dim) x (embed_dim x embed_dim) ==> (seq_length x embed_dim)
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            # 如果q,k,v的最后一维是一致的，则组成一个大的权重矩阵
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        # 之后会将所有的头注意力拼接在一起，然后乘上权重矩阵输出
        # out_proj是为了后期准备的
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            # 从连续型均匀分布里面随机取样作为初始值
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            # 进行全零向量初始化
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
```


```python
src = torch.randn((2,4,100))
src = positional_encoding(src,100,0.1)
print(src.shape)
multihead_attn = MultiheadAttention(100, 4, 0.1)
attn_output, attn_output_weights = multihead_attn(src,src,src)
print(attn_output.shape, attn_output_weights.shape)
```

    torch.Size([2, 4, 100])
    torch.Size([2, 4, 100]) torch.Size([4, 2, 2])
    

#### 2.3.4 TransformerEncoderLayer


```python
class TransformerEncoderLayer(nn.Module):
    """
    参数：
        d_model: 词嵌入的维度（必备）
        nhead: 多头注意力中平行头的数目（必备）
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率（Default = 0.1）
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0（Default = 1e-5）
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)（Default：False）
    """

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src = positional_encoding(src, src.shape[-1])
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src
```


```python
encoder_layer = TransformerEncoderLayer(d_model=512, n_head=8)
src = torch.randn((32, 10, 512))
out = encoder_layer(src)
print(out.shape)
```

    torch.Size([32, 10, 512])
    

#### 2.3.5 Transformer Layer组成Encoder


```python
class TransformerEncoder(nn.Module):
    """
    参数：
        encoder_layer（必备）
        num_layers： encoder_layer的层数（必备）
        norm: 归一化的选择（可选）
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = positional_encoding(src, src.shape[-1])
        for _ in range(self.num_layers):
            output = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
```


```python
encoder_layer = TransformerEncoderLayer(d_model=512, n_head=8)
transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
src = torch.randn((10, 32, 512))
out = transformer_encoder(src)
print(out.shape)
```

    torch.Size([10, 32, 512])
    

#### 2.3.6 Decoder Layer


```python
class TransformerDecoderLayer(nn.Module):
    """
    参数：
        d_model: 词嵌入的维度（必备）
        nhead: 多头注意力中平行头的数目（必备）
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率（Default = 0.1）
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0（Default = 1e-5）
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)（Default：False）
    """

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        参数：
            tgt: 目标语言序列（必备）
            memory: 从最后一个encoder_layer跑出的句子（必备）
            tgt_mask: 目标语言序列的mask（可选）
            memory_mask（可选）
            tgt_key_padding_mask（可选）
            memory_key_padding_mask（可选）
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
```


```python
decoder_layer = TransformerDecoderLayer(d_model=512, n_head=8)
memory = torch.randn((10, 32, 512))
tgt = torch.randn((20, 32, 512))
out = decoder_layer(tgt, memory)
print(out.shape)
```

    torch.Size([20, 32, 512])
    

#### 2.3.7 Decoder


```python
class TransformerDecoder(nn.Module):
    r"""
    参数：
        decoder_layer（必备）
        num_layers: decoder_layer的层数（必备）
        norm: 归一化选择
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for _ in range(self.num_layers):
            output = self.layer(output, memory, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output
```


```python
decoder_layer =TransformerDecoderLayer(d_model=512, n_head=8)
transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
print(out.shape)
```

    torch.Size([20, 32, 512])
    

#### 2.3.8 Transformer


```python
class Transformer(nn.Module):
    """
    参数：
        d_model: 词嵌入的维度（必备）（Default=512）
        nhead: （必备）（Default=8）
        num_encoder_layers:编码层层数（Default=8）
        num_decoder_layers:解码层层数（Default=8）
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率（Default = 0.1）
        activation: 两个线性层中间的激活函数，默认relu或gelu
        custom_encoder: 自定义encoder（Default=None）
        custom_decoder: 自定义decoder（Default=None）
        lay_norm_eps: layer normalization中的微小量，防止分母为0（Default = 1e-5）
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)（Default：False）
    """

    def __init__(self, d_model: int = 512, n_head: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False) -> None:
        super(Transformer, self).__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first)
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first)
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.n_head = n_head

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        参数：
            src: 源语言序列（送入Encoder）（必备）
            tgt: 目标语言序列（送入Decoder）（必备）
            src_mask: （可选)
            tgt_mask: （可选）
            memory_mask: （可选）
            src_key_padding_mask: （可选）
            tgt_key_padding_mask: （可选）
            memory_key_padding_mask: （可选）

        形状：
            - src: shape:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: shape:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: shape:`(S, S)`.
            - tgt_mask: shape:`(T, T)`.
            - memory_mask: shape:`(T, S)`.
            - src_key_padding_mask: shape:`(N, S)`.
            - tgt_key_padding_mask: shape:`(N, T)`.
            - memory_key_padding_mask: shape:`(N, S)`.

            [src/tgt/memory]_mask确保有些位置不被看到，如做decode的时候，只能看该位置及其以前的，而不能看后面的。
            若为ByteTensor，非0的位置会被忽略不做注意力；若为BoolTensor，True对应的位置会被忽略；
            若为数值，则会直接加到attn_weights

            [src/tgt/memory]_key_padding_mask 使得key里面的某些元素不参与attention计算，三种情况同上

            - output: shape:`(T, N, E)`, `(N, T, E)` if batch_first.

        注意：
            src和tgt的最后一维需要等于d_model，batch的那一维需要相等

        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """产生关于序列的mask，被遮住的区域赋值`-inf`，未被遮住的区域赋值为`0`"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """用正态分布初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
```


```python
transformer_model = Transformer(n_head=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
print(out.shape)
```

    torch.Size([20, 32, 512])
    

## 3 总结

&emsp;&emsp;本次任务，主要从seq2seq模型，到Attention模型，再到Transformer模型，并详细介绍了Transformer的各个组成部分，包括Transformer输入、Encoder编码器、Self-Attention、残差连接、Decoder解码器、最后的线性层和softmax层。根据Transformer原理，详细讲解了Transformer代码。
