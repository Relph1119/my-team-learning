# Task01 PyTorch简介与基础知识

## 1 PyTorch简介

- 概念：由Facebook人工智能研究小组开发的一种基于Lua编写的Torch库的Python实现的深度学习库
- 优势：简洁、上手快、具有良好的文档和社区支持、项目开源、支持代码调试、丰富的扩展库

## 2 PyTorch基础知识

### 2.1 张量

- 分类：0维张量（标量）、1维张量（向量）、2维张量（矩阵）、3维张量（时间序列）、4维张量（图像）、5维张量（视频）
- 概念：一个数据容器，可以包含数据、字符串等


```python
import torch
```


```python
# 创建tensor
x = torch.rand(4, 3)
print(x)
# 构造数据类型为long，数据是0的矩阵
x = torch.zeros(4, 3, dtype=torch.long)
print(x)
```

    tensor([[0.9515, 0.6332, 0.8228],
            [0.3508, 0.0493, 0.7606],
            [0.7326, 0.7003, 0.1925],
            [0.1172, 0.8946, 0.9501]])
    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    

- 常见的构造Tensor的函数：
|                  函数 | 功能                             |
| --------------------:| ------------------------------------------------------- |
| Tensor(\**sizes*)    | 基础构造函数                             |
| tensor(*data*)      | 类似于np.array                            |
| ones(\**sizes*)      | 全1                                   |
| zeros(\**sizes*)    | 全0                                    |
| eye(\**sizes*)      | 对角为1，其余为0                           |
| arange(*s,e,step*)   | 从s到e，步长为step                         |
| linspace(*s,e,steps*) | 从s到e，均匀分成step份                       |
| rand/randn(\**sizes*) | rand是\[0,1)均匀分布；randn是服从N（0，1）的正态分布   |
| normal(*mean,std*)   | 正态分布（均值为mean，标准差是std）              |
| randperm(*m*)      | 随机排列                                 |

- 操作：
  1. 使用索引表示的变量与原数据共享内存，即修改其中一个，另一个也会被修改
  2. 使用`torch.view`改变tensor的大小
  3. 广播机制：当对两个形状不同的Tensor按元素运算时，可能会触发广播(broadcasting)机制


```python
# 使用view改变张量的大小
x = torch.randn(5, 4)
y = x.view(20)
z = x.view(-1, 5) # -1是指这一维的维数由其他维度决定
print(x.size(), y.size(), z.size())
```

    torch.Size([5, 4]) torch.Size([20]) torch.Size([4, 5])
    


```python
# 广播机制
x = torch.arange(1, 3).view(1, 2)
y = torch.arange(1, 4).view(3, 1)
print("x =", x)
print("y =", y)
print("x + y =", x + y)
```

    x = tensor([[1, 2]])
    y = tensor([[1],
            [2],
            [3]])
    x + y = tensor([[2, 3],
            [3, 4],
            [4, 5]])
    

### 2.2 自动求导

- `autograd`包：提供张量上的自动求导机制
- 原理：如果设置`.requires_grad`为`True`，那么将会追踪张量的所有操作。当完成计算后，可以通过调用`.backward()`自动计算所有的梯度。张量的所有梯度将会自动累加到`.grad`属性
- `Function`：`Tensor`和`Function`互相连接生成了一个无环图 (acyclic graph)，它编码了完整的计算历史。每个张量都有一个`.grad_fn`属性，该属性引用了创建`Tensor`自身的`Function`


```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    


```python
y = x ** 2
print(y)
```

    tensor([[1., 1.],
            [1., 1.]], grad_fn=<PowBackward0>)
    


```python
z = y * y * 3
out = z.mean()
print("z = ", z)
print("z mean = ", out)
```

    z =  tensor([[3., 3.],
            [3., 3.]], grad_fn=<MulBackward0>)
    z mean =  tensor(3., grad_fn=<MeanBackward0>)
    

- 梯度：对于那么 $\vec{y}$ 关于 $\vec{x}$ 的梯度就是一个雅可比矩阵
$$
J=\left(
\begin{array}{ccc}
\frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}} \\ 
\vdots & \ddots & \vdots \\ 
\frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
\end{array}\right)
$$

- `grad`的反向传播：运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零


```python
out.backward()
print(x.grad)
```

    tensor([[3., 3.],
            [3., 3.]])
    


```python
# 反向传播累加
out2 = x.sum()
out2.backward()
print(x.grad)
```

    tensor([[4., 4.],
            [4., 4.]])
    

### 2.3 并行计算

- 目的：通过使用多个GPU参与训练，加快训练速度，提高模型学习的效果
- CUDA：通过使用NVIDIA提供的GPU并行计算框架，采用`cuda()`方法，让模型或者数据迁移到GPU中进行计算

- 并行计算方法：
  1. Network partitioning：将一个模型网络的各部分拆分，分配到不同的GPU中,执行不同的计算任务
  2. Layer-wise partitioning：将同一层模型拆分，分配到不同的GPU中，训练同一层模型的部分任务
  3. Data parallelism（主流）：将不同的数据分配到不同的GPU中，执行相同的任务

## 3 总结

&emsp;&emsp;本次任务，主要介绍了PyTorch概念及优势、以及基础知识，包括张量、自动求导和并行计算；通过构建张量，存储我们需要的数据；基于自动求导机制和雅可比矩阵的计算规则，计算张量的梯度；并行计算方法主要包括Network partitioning、Layer-wise partitioning和Data parallelism，目前主流的是最后一种。
