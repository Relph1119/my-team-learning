# Task05 网络设计的技巧

## 1 局部最小值与鞍的处理

### 1.1 优化失败的原因

当参数的微分为0的时候，梯度下降无法继续更新参数，导致训练损失误差无法下降
- 卡在了局部最小值（`Local maxima`）
- 卡在了鞍点（`saddle point`）

### 1.2 判断局部最小值与鞍点的方法

1. 得到损失函数的形状  
给定某一组参数$\theta=\theta'$，对$L(\theta)$进行泰勒展开
$$
L(\theta) \approx L(\theta') + (\theta - \theta')^T g + \frac{1}{2}(\theta - \theta')^T H (\theta - \theta')
$$
其中$g$表示$L(\theta')$对$\theta$梯度：
$$
g = \nabla L(\theta') \quad g_i = \frac{\partial L(\theta')}{\partial \theta_i}
$$
$H$表示是$L(\theta')$的二次微分（即Hession矩阵）：
$$
H_ij = \frac{\partial^2}{\partial \theta_i \partial \theta_j} L(\theta')
$$

2. 如果是`critical point`（临界点），则$(\theta - \theta')^T g = 0$，会出现以下3种情况：  
记$v = \theta - \theta'$，考虑$(\theta - \theta')^T H (\theta - \theta') = v^T H v$  
（1）当$v^T H v > 0$时，即$H$为正定矩阵时（所有特征都是正值），$L(\theta) > L(\theta')$，则是局部最小值  
（2）当$v^T H v < 0$时，即$H$为非正定矩阵时（所有特征都是负值），$L(\theta) < L(\theta')$，则是局部最小值  
（3）当有$v^T H v > 0$或$v^T H v < 0$时，即特征有正有负，则是鞍点  

### 1.3 $H$矩阵可以指导参数更新的方向

1. 当在临界点时，即$\displaystyle L(\theta) \approx L(\theta')+ \frac{1}{2}(\theta - \theta')^T H (\theta - \theta')$  
2. 假设位于鞍点处，设$u$是$H$的特征子矩阵，$\lambda$是$u$的一个特征值，可得$u^T H u = u^T (\lambda u) = \lambda \|u\|^2 < 0$  
3. 将$\theta - \theta' = u$代入$L(\theta)$式中，得到$L(\theta) < L(\theta')$，则更新$\theta = \theta' + u$，可以继续减小$L(\theta)$

## 2 批次（Batch）与动量（Momentum）

### 2.1 Batch

- 定义：将整个训练集分为多个`Batch`，依次对这些`Batch`计算梯度并更新参数，全部更新一遍称为一个`epoch`，需要先进行`Shuffle`

- Small Batch v.s. Large Batch：  
（1）当batch size很大时，并不需要很长的时间计算梯度（除非batch size太大了）  
（2）很小的batch在一个epoch上需要耗费更多的时间（遍历所有的数据）  
（3）小的batch会使得模型有更好的效果

|| Small | Large |
| :-- | :--: | :--: |
| Seppd for one update(no parallel) | Faster | Slower |
| Speed for one update(with parallel) | Same| Same(not too large) |
| Time for one epoch | Slower | Faster |
| Gradient | Noisy | Stable |
| Optimization | Better | Worse |
| Generalization | Better | Worse |

### 2.2 动量（Momentum）

- 一般的梯度下降：往梯度的反方向更新参数，$\theta^{(i+1)} = \theta^{(i)} - \eta g^{(i)}$
- 动量的梯度下降：加上动量在进行参数更新，$m^{(i+1)} = \lambda m^{(i)} - \eta g^{(i)}, \theta^{(i+1)} = \theta^{(i)} + m^{(i+1)}$

## 3 自动调整学习率

1. 假设只考虑一个参数$\theta$，可知参数更新如下：
$$
\theta_i^{t+1} \leftarrow \theta_i^t - \eta g_i^t
$$
其中$\displaystyle g_i^t = \frac{\partial L}{ \partial \theta_i} |_{\theta=\theta^t}$  
2. 使用特质化的学习率
$$
\theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta}{\sigma_i^t} g_i^t
$$

- Root Mean Square：$\displaystyle \sigma_i^t = \sqrt{\frac{1}{t+1} \sum_{i=0}^t (g_i^t)^2}$

- RMSProp：$\sigma_i^t = \sqrt{\alpha (\sigma_i^{t-1})^2 + (1 - \alpha) (g_i^t)^2}$

3. 学习率调度（Learning Rate Scheduling）
$$\displaystyle \theta_i^{t+1} \leftarrow \theta_i^t - \frac{\eta^t}{\sigma_i^t} g_i^t$$
- Learning Rate Decay：在训练开始之后，逐步接近目标，所以$\eta^t$随着$t$逐渐变小
- Warm Up：在训练开始，$\sigma_i^t$统计比较精准之后，再逐步接近目标，所以$\eta^t$随着$t$先变大，再逐渐变小

## 4 损失函数的影响

- 回归（Regression）：
$$
\hat{y} \leftrightarrow y = b + c^T \sigma(b + Wx)
$$

- 分类（Classification）：
$$
y = b' + W' \sigma(b + Wx) \\
\hat{y} \leftrightarrow y' = \text{softmax}(y) \\
y'_i = \frac{\exp (y_i)}{\displaystyle \sum_j \exp(y_i)}
$$

- 分类的损失函数：
$$
L = \frac{1}{N} \sum_n e_n
$$
其中，如果采用MSE（Mean Square Error），则$\displaystyle e = \sum_i (\hat{y}_i - y'_i)^2$  
如果采用交叉熵（Cross-entropy），则$\displaystyle e = -\sum_i \hat{y}_i \ln y'_i$，这个更常用。

## 5 批次标准化（Batch Normalization）

- 应用场景：在不同的数据值（很小、很大），会导致学习率不一致，在数值很小的特征中，学习路径很平坦，在数值相差很大的特征中，学习路径很陡。

- 特征标准化：通常情况下，使得梯度下降更快
$$
\tilde{x}_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}
$$
其中，$i$表示维度，$m_i$表示第$i$维度的平均值，$\sigma_i$表示标准差

- 批次标准化：
$$
\tilde{z}^i = \frac{z_i - u}{\sigma} \\
\hat{z}^i = \gamma \odot \tilde{z}^i + \beta
$$

- 内部协变量偏移(Internal Covariate Shift)：函数梯度变化成线性增长，不能保证对数据的非线性变换，从而影响数据表征能力，降低神经网络的作用。

## 6 总结

&emsp;&emsp;本次任务，主要介绍了在设计网络中可以使用的一些技巧：
1. 优化失败的主要原因来自于局部最小值和鞍点，通过对$H$的判断，可以识别训练卡在了局部最小值，还是卡在了鞍点
2. 使用Batch能加快梯度的计算，添加动量能更好的越过局部最小值或鞍点
3. 利用RMS和RMSProp自动调整学习率，并使用学习率调度，加快梯度的下降速度
4. 使用交叉熵能更好的计算损失
5. 使用批次标准化，能解决各个输入数据的偏差不一致的情况
