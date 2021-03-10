# Task02 基于统计学的方法 {docsify-ignore-all}

## 1 知识梳理

### 1.1 概述
- 统计学方法：假定正常的数据对象由一个统计模型产生，而不遵守该模型的数据则是异常点。
- 一般思想：学习一个拟合给定数据集的生成模型，然后识别该模型低概率区域中的对象，把它们作为异常点
- 两个主要类型：参数方法和非参数方法

### 1.2 参数方法
- 基于正态分布的一元异常点检测：   
&emsp;&emsp;根据输入数据集，其样本服从正态分布$x^{(i)}\sim N(\mu, \sigma^2)$，可求出参数$\mu$和$\sigma$  
$$\mu=\frac 1m\sum_{i=1}^m x^{(i)} \\ \sigma^2=\frac 1m\sum_{i=1}^m (x^{(i)}-\mu)^2$$
&emsp;&emsp;故概率密度函数为$\displaystyle p(x)=\frac 1{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$  
&emsp;&emsp;通常采用$3\sigma$原则，如果数据点超过范围$(\mu-3\sigma, \mu+3\sigma)$，那么这些点很有可能是异常点

- 多元异常点检测：  
&emsp;&emsp;假设各个维度的特征之间相互独立，对于第$j$维有：$$\mu_j=\frac 1m\sum_{i=1}^m x_j^{(i)} \\ \sigma_j^2=\frac 1m\sum_{i=1}^m (x_j^{(i)}-\mu_j)^2$$其概率密度函数为$\displaystyle p(x)=\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n\frac 1{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})$

- 特征之间有相关性，且符合多元高斯分布：
$$\mu=\frac{1}{m}\sum^m_{i=1}x^{(i)} \\ \sum=\frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu)(x^{(i)}-\mu)^T$$其概率密度函数为$\displaystyle p(x)=\frac{1}{(2 \pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)$

### 1.3 非参数方法
使用直方图进行异常检测：
1. 构造直方图，指定直方图类型、箱数、每个箱的大小
2. 对照直方图，检测异常点

缺点：难以选择一个合适的箱大小

### 1.4 HBOS方法
- 概念：假设数据集中的每个维度相互独立，对各维度进行区间(bin)划分，区间的密度越高，异常评分越低。
- 算法流程：  
    1. 为每个维度构造直方图：静态宽度直方图、动态宽带直方图
    2. 计算各维度的直方图，其中每个箱子的高度表示密度估计，将直方图进行归一化，计算各直方图的`HBOS`值：
$$
\text{HBOS}(p)=\sum_{i=0}^{d} \log \left(\frac{1}{\text {hist}_{i}(p)}\right)
$$

## 2 练习
使用`PyOD`库生成`toy example`并调用`HBOS`
