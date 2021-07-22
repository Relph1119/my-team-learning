# Task05 详读西瓜书+南瓜书第6章

## 1 间隔与支持向量
- 支持向量的概念：  
  &emsp;&emsp;假设超平面$(w,b)$能将训练样本正确分类，即对于$(x_i,y_i) \in D$，若$y_i=+1$，则有$w^Tx_i+b>0$，若$y_i=-1$，则有$w^Tx_i+b<0$，令：
  $$\left \{ 
  \begin{array}{ccc}
   w^Tx_i+b \geqslant +1, \quad y_i=+1 \\
   w^Tx_i+b \leqslant -1, \quad y_i=-1
  \end{array} \right .$$
  距离超平面最近的这几个训练样本点使得上式成立，则这些样本点被称为支持向量。
- 间隔：两个异类支持向量超平面的距离之和$\displaystyle \gamma=\frac{2}{\|w\|}$
- 支持向量机（SVM）基本型：  
  &emsp;&emsp;找到最大间隔的划分超平面，需要求解参数$w$和$b$使得$\gamma$最大，目标函数如下：$$\begin{array}{l} \min \limits_{w,b} & \displaystyle \frac{1}{2}\|w\|^2 \\ 
\text { s.t. } & y_{i}(w^T x_i+b ) \geqslant 1, \quad i=1,2, \ldots, m \end{array}$$

## 2 对偶问题
- 拉格朗日函数：$\displaystyle L(w, b, \alpha)=\frac{1}{2}\|w\|^2 + \sum_{i=1}^m \alpha_i\left(1- y_i (w^T x_i+b ) \right)$
- 对偶问题：$$\begin{array}{ll} \displaystyle \max_{\alpha} & \displaystyle \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j \\ 
{\text { s.t. }} & {\displaystyle \sum_{i=1}^m \alpha_i y_i=0} \\ 
{} & {\alpha_i \geqslant 0, i=1,\ldots, m}
\end{array}$$
- KKT条件：$$\left \{ \begin{array}{ll}
\alpha_i \geqslant 0 \\
y_i f(x_i) - 1 \geqslant 0 \\
\alpha_i(y_i f(x_i) - 1) = 0
\end{array}\right .$$
- 支持向量机特点：训练完成后，大部分的训练样本都不需保留，最终模型仅与支持向量有关
- SMO算法思路：
  1. 选取一对需要更新的变量$\alpha_i$和$\alpha_j$
  2. 固定$\alpha_i$和$\alpha_j$以外的参数，求解对偶问题，获得更新后的$\alpha_i$和$\alpha_j$
  3. 重复上述2个步骤，直至收敛
- SMO采用一个启发式：使选取的两变量所对应样本之间的间隔最大

## 3 软间隔与正则化
- 软间隔：允许某项样本不满足约束$y_i(w^Tx_i+b) \geqslant 1$，在最大化间隔的同时，不满足约束的样本应该尽可能少
- 目标函数：$$\begin{array}{l}
\displaystyle \min_{w, b} & \displaystyle \frac{1}{2}\|w\|^2+C \sum_{i=1}^m \ell_{0/1} \left( y_i(w^T x_i+b) -1 \right) \\ 
& {\ell_{0/1}(z)=\left\{\begin{array}{ll}
{1,} & {\text { if } z<0 ;} \\ 
{0,} & {\text { otherwise }} \end{array}\right.}
\end{array}$$
- 损失函数：
  1. hinge损失：$\ell_{hinge}(z)=\max(0,1-z)$  
  2. 指数损失（exponential loss）: $\ell_{exp}(z)=\exp (-z)$  
  3. 对率损失（logistic loss）：$\ell_{log}(z)=log(1+\exp(-z))$
- 常用的软间隔支持向量机：
$$\begin{array}{ll}
\displaystyle \min \limits_{w,b,\xi} & \displaystyle \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \xi_i \\ 
\text{s.t.} & y_i (w^T x_i+b) \geqslant 1-\xi_i \\
& \xi_i \geqslant 0, \quad i=1, \ldots, m 
\end{array}$$
- 软间隔支持向量机的对偶问题：
$$\begin{array}{cl}
{\displaystyle \max \limits_{\alpha}} & {\displaystyle \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j} \\ 
{\text {s.t.}} & {\displaystyle \sum_{i=1}^m \alpha_i y_i=0}  \\
{} & {0 \leqslant \alpha_i \leqslant C, i=1,2, \ldots, m}
\end{array}$$
- 软间隔支持向量机的KKT条件：
$$\left \{ \begin{array}{l}
\alpha_i \geqslant 0, \quad \mu_i \geqslant 0 \\
y_i f(x_i) - 1 + \xi_i \geqslant 0 \\
\alpha_i (y_i f(x_i) - 1 + \xi_i) = 0 \\
\xi_i \geqslant 0, \quad \mu_i \xi_i = 0
\end{array}\right .$$
- 软间隔支持向量机的最终模型仅与支持向量有关，即通过采用hinge损失函数仍保持了稀疏性
- 正则化问题：$$\min \limits_{f} \Omega(f) + C \sum_{i=1}^m \ell(f(x_i), y_i)$$在该式中，$\Omega(f)$称为正则化项，$C$称为正则化参数
  1. $L_p$范数使常用的正则化项
  2. $L_2$范数$\|w\|_2$倾向于$w$的分量取值尽量均衡，即非零分量个数尽量稠密
  3. $L_0$范数$\|w\|_0$和$L_1$范数$\|w\|_1$倾向于$w$的分量尽量系数，即非零分量个数尽量少

## 4 支持向量回归
- 损失计算规则：以$f(x)$为中心，构建一个宽度为$2\epsilon$的间隔带，若训练样本落入此间隔带，则不计算损失，认为是被预测正确
- SVR目标函数：$$\begin{array}{l}
\displaystyle \min_{w, b, \xi_i,\hat{\xi}_i} & \displaystyle \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m (\xi_i + \hat{\xi}_i) \\ 
\text{s.t.} & f(x_i) - y_i \leqslant \epsilon + \xi_i \\
& y_i - f(x_i) \leqslant \epsilon + \hat{\xi}_i \\
& \xi_i \geqslant 0, \hat{\xi}_i \geqslant 0, i=1,2,\ldots,m
\end{array}$$
- SVR对偶问题：
$$\begin{array}{l}
\max \limits_{\alpha, \hat{\alpha}} &\displaystyle \sum_{i=1}^{m} y_{i}(\hat{\alpha}_{i}-\alpha_{i})-\epsilon(\hat{\alpha}_{i}+\alpha_{i}) -\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} (\hat{\alpha}_{i}-\alpha_{i})(\hat{\alpha}_{j}-\alpha_{j}) x_i^T x_j \\
\text { s.t. } &\displaystyle \sum_{i=1}^{m} (\hat{\alpha}_{i}-\alpha_{i})=0 \\
& 0 \leqslant \alpha_{i}, \hat{\alpha}_{i} \leqslant C
\end{array}$$
- SVR的KKT条件：
$$\left\{\begin{array}{l}
\alpha_i (f(x_i) - y_i - \epsilon - \xi_i)=0 \\
\hat{\alpha}_i (y_i -f(x_i) - \epsilon - \hat{\xi}_i)=0 \\
\alpha_i \hat{\alpha}_i=0, \quad \xi_i \hat{\xi}_i=0 \\
(C-\alpha_i ) \xi_i=0, \quad (C-\hat{\alpha}_i) \hat{\xi}_i=0
\end{array}\right.$$
- SVR的解：$$f(x)=\sum_{i=1}^m (\hat{\alpha}_i - \alpha_i) x_i^T x + b
$$其中：$$b = y_i + \epsilon - \sum_{i=1}^m (\hat{\alpha}_i - \alpha_i) x_i^T x \\
w = \sum_{i=1}^m (\hat{\alpha}_i - \alpha_i ) \phi (x_i)$$
