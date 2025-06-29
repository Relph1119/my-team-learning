# Task02 Transformer架构

## 1 注意力机制

- 核心架构：前馈神经网络（FNN）、卷积神经网络（CNN）、循环神经网络（RNN）
- 注意力机制：将重点注意力集中在一个或几个 token，从而取得更高效高质的计算效果。
- 注意力机制的核心变量：Query（查询值）、Key（键值）和 Value（真值）；当Key与Query相关性越高，则其所应该赋予的注意力权重就越大。

**注意力机制公式推导：**

1. 用$ v·w =\displaystyle \sum_{i}v_iw_i $ 公式表示词向量的相似性，语义相似，这个值就大于0，语义不相似，这个值就小于0。
2. 计算查询值Query与字典中每个Key的相似度：$ x=q K^T $。
3. 将上述相似度进行归一化：$ \text{softmax}(x)_i = \displaystyle \frac{e^{xi}}{\sum_{j}e^{x_j}} $。
4. 得到注意力的基本公式：$ \text{attention}(Q,K,V) = \text{softmax}(qK^T)v $。
5. 一次性查询多个Query，用Q表示，即注意力的基本公式为$\text{attention}(Q,K,V) = \text{softmax}(QK^T) V $。
6. 对Q和K的乘积进行缩放，得到最终的注意力公式：$\text{attention}(Q,K,V) = \text{softmax} (\frac{QK^T}{\sqrt{d_k}}) V $。

**注意力机制的代码实现：**


```python
import math
import torch

def attention(query, key, value):
    """
    注意力计算函数
    :param query: 查询值矩阵
    :param key: 键值矩阵
    :param value: 真值矩阵
    :return:
    """
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并除以根号dk
    # transpose相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn
```

- 自注意力：计算本身序列中每个元素对其他元素的注意力分布。在具体实现中是在attention函数中，通过给 Q、K、V 的输入传入同一个参数实现的。
```python
attention(Q,Q,Q)
attention(K,K,K)
attention(V,V,V)
```

- 掩码自注意力：使用注意力掩码（遮蔽一些特定位置的 token）的自注意力机制。其目的是让模型只能使用历史信息进行预测而不能看到未来信息。掩码矩阵是一个和文本序列等长的上三角矩阵。当输入维度为 （batch_size, seq_len, hidden_size）时，掩码矩阵维度一般为 (1, seq_len, seq_len)。

- 多头注意力：将原始的输入序列进行多组的自注意力处理，然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理，得到最终的输出。

**多头注意力的代码实现：**


```python
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    """模型参数"""
    n_embd: int  # 嵌入维度
    n_heads: int  # 头数
    dim: int  # 模型维度
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int
    n_layer: int


class MultiHeadAttention(nn.Module):
    """多头自注意力计算模块"""

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.n_embd % args.n_heads == 0
        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 n_embd x n_embd（head_dim = n_embeds / n_heads）
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal
        
        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seq_len, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seq_len, :seq_len]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

## 2 Encoder-Decoder

- Seq2Seq 模型：序列到序列，对自然语言序列进行编码再解码。

- 前馈神经网络（FFN）：每一层的神经元都和上下两层的每一个神经元完全连接的网络结构。


```python
class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 输入x通过第一层线性变换和RELU激活函数
        # 通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
```

- 层归一化：将深度神经网络中每一层的输入归一化成标准正态分布。
    1. 计算样本的均值： $\displaystyle \mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^{i}$
    2. 计算样本的方差：$\displaystyle \sigma^2 = \frac{1}{m}\sum^{m}_{i=1}(Z_j^i - \mu_j)^2$
    3. 进行归一化：$\displaystyle \widetilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}$


```python
class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 线性矩阵做映射
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 在统计每个样本所有维度的值，求均值和方差
        mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

- 残差连接：目的是为了避免模型退化，允许最底层信息直接传到最高层。

- Encoder实现：由 N 个 Encoder Layer 组成，每一个 Encoder Layer 包括一个注意力层和一个前馈神经网络。


```python
class EncoderLayer(nn.Module):
    """Encoder层"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out


class Encoder(nn.Module):
    """Encoder 块"""

    def __init__(self, args: ModelArgs):
        super(Encoder, self).__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        # 分别通过 N 层 Encoder Layer
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

- Decoder：由两个注意力层和一个前馈神经网络组成。第一个注意力层是一个掩码自注意力层，第二个注意力层是一个多头注意力层，该层将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。最后，再经过前馈神经网络。


```python
class DecoderLayer(nn.Module):
    """Decoder层"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(x, x, x)
        # 多头注意力
        x = self.attention_norm_2(x)
        h = x + self.attention.forward(x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, args: ModelArgs):
        super(Decoder, self).__init__()
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        # 分别通过每一个Decoder Layer
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```

## 3 搭建一个 Transformer

- Embeddng 层：存储固定大小的词典的嵌入向量查找表。让自然语言输入通过分词器 tokenizer，形成一个固定的词序码。

- 位置编码：保留词在语句序列中的相对位置信息，Transformer使用了正余弦函数来进行位置编码。

$$
\text{PE} (\text{pos}, 2i) = sin(\frac{\text{pos}} {10000^{2i/d_{\text{model}}}}) \\
\text{PE} (\text{pos}, 2i+1) = cos(\frac{\text{pos}} {10000^{2i/d_{\text{model}}}})
$$


```python
import numpy as np
import matplotlib.pyplot as plt

def PositionEncoding(seq_len, d_model, n=10000):
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
```


```python
P = PositionEncoding(seq_len=4, d_model=4, n=100)
print(P)
```

    [[ 0.          1.          0.          1.        ]
     [ 0.84147098  0.54030231  0.09983342  0.99500417]
     [ 0.90929743 -0.41614684  0.19866933  0.98006658]
     [ 0.14112001 -0.9899925   0.29552021  0.95533649]]
    

按照torch的规则，依据位置编码的实现原理，代码实现如下：


```python
class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, args: ModelArgs):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

- 一个完整的 Transformer：经过 tokenizer 映射后的输出，先经过 Embedding 层和 Positional Embedding 层编码，然后进入 N 个 Encoder 和 N 个 Decoder（在 Transformer 原模型中，N 取为6），最后经过一个线性层和一个 Softmax 层就得到了最终输出。


```python
class Transformer(nn.Module):
    """Transformer整体模型"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            wpe=PositionalEncoding(args),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=False):
        """统计所有参数的数量"""
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """初始化权重"""
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """前向计算函数"""
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx:", idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb:", tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb)
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:", x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:", enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:", x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```

测试一下这个Transformer是否能正常运行：


```python
from transformers import BertTokenizer

args = ModelArgs(100, 10, 100, 0.1, 512, 1000, 1000, 2)

text = "我喜欢快乐地学习大模型"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
inputs_token = tokenizer(
    text,
    return_tensors='pt',
    max_length=args.max_seq_len,
    truncation=True,
    padding='max_length'
)
args.vocab_size = tokenizer.vocab_size
transformer = Transformer(args)
inputs_id = inputs_token['input_ids']
logits, loss = transformer.forward(inputs_id)
print("logits:", logits)
predicted_ids = torch.argmax(logits, dim=-1).item()
output = tokenizer.decode(predicted_ids)
print("output:", output)
```

    number of parameters: 4.55M
    idx: torch.Size([1, 512])
    tok_emb: torch.Size([1, 512, 100])
    x after wpe: torch.Size([1, 512, 100])
    enc_out: torch.Size([1, 512, 100])
    x after decoder: torch.Size([1, 512, 100])
    logits: tensor([[[ 0.1250,  0.1328,  0.0184,  ..., -0.0308,  0.0209,  0.1302]]],
           grad_fn=<UnsafeViewBackward0>)
    output: ᅥ
    
