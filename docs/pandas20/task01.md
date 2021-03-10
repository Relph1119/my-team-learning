# Task01 预备知识 {docsify-ignore-all}

## 1 知识梳理（重点记忆）

### 1.1 np数组构造

特殊数组的生成方式


```python
import numpy as np
```


```python
target = np.arange(1,13).reshape(2,6)
target
```


```python
# 按行填充
target.reshape((6,2), order='C')
```


```python
# 按列填充
target.reshape((3,4), order='F')
```

### 1.2 向量与矩阵的计算
向量内积


```python
a = np.array([1,2,4])
b = np.array([2,3,4])
a.dot(b)
```

矩阵外积


```python
np.outer(a, b)
```


```python
np.outer(b, a)
```

矩阵乘法


```python
a = np.arange(4).reshape(-1, 2)
a
```


```python
b = np.arange(4).reshape(2, -1)
b
```


```python
a@b
```

## 2 练习

### 2.1 Ex1：利用列表推导式写矩阵乘法
一般的矩阵乘法根据公式，可以由三重循环写出，请将其改写为列表推导式的形式。


```python
M1 = np.random.rand(2,3)
M2 = np.random.rand(3,4)
res = np.empty((M1.shape[0],M2.shape[1]))
for i in range(M1.shape[0]):
    for j in range(M2.shape[1]):
        item = 0
        for k in range(M1.shape[1]):
            item += M1[i][k] * M2[k][j]
        res[i][j] = item        
(abs(M1@M2 - res) < 1e-15).all() # 排除数值误差
```

**我的解答：**  
思路分析：  
1. 先将最内层(k层)`for`循环改成列表推导形式
```python
sum([M1[i][k]* M2[k][j] for k in range(M1.shape[1])])
```

2. 再将第二层(j层)`for`循环进行整合
```python
[sum([M1[i][k]* M2[k][j] for k in range(M1.shape[1])]) for j in range(M2.shape[1])]
```

3. 最后再整合第一层`for`循环进行整合


```python
res = [[sum([M1[i][k]* M2[k][j] for k in range(M1.shape[1])]) for j in range(M2.shape[1])] for i in range(M1.shape[0])]
```


```python
(abs(M1@M2 - res) < 1e-15).all()
```

### 2.2 Ex2：更新矩阵
设矩阵 $A_{m×n}$ ，现在对 $A$ 中的每一个元素进行更新生成矩阵 $B$ ，更新方法是 $B_{ij}=A_{ij}\sum_{k=1}^n\frac{1}{A_{ik}}$ ，例如下面的矩阵为 $A$ ，则 $B_{2,2}=5\times(\frac{1}{4}+\frac{1}{5}+\frac{1}{6})=\frac{37}{12}$ ，请利用 `Numpy` 高效实现。
$$A=\left[ \begin{array}{ccc} 
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9 
\end{array}\right]$$

**我的解答：**


```python
A = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])
```

1. 首先计算$\displaystyle \sum_{k=1}^n\frac{1}{A_{ik}}$


```python
C = (1./A).sum(1)
C
```

由于得到的是行向量，需要使用`reshape(-1,1)`转换成列向量


```python
C = C.reshape(-1,1)
C
```

2. 再计算$B=A\times C$


```python
B = A * C
```


```python
assert B[1,2] == 6*(1/4+1/5+1/6)
assert B[1,1] == 37/12
```

### 2.3 Ex3：卡方统计量

设矩阵$A_{m\times n}$，记$\displaystyle B_{ij} = \frac{(\sum_{i=1}^mA_{ij})\times (\sum_{j=1}^nA_{ij})}{\sum_{i=1}^m\sum_{j=1}^nA_{ij}}$，定义卡方值如下：
$$\chi^2 = \sum_{i=1}^m\sum_{j=1}^n\frac{(A_{ij}-B_{ij})^2}{B_{ij}}$$
请利用`Numpy`对给定的矩阵$A$计算$\chi^2$ 


```python
np.random.seed(0)
A = np.random.randint(10, 20, (8, 5))
```

**我的解答：**  
使用外连接构造矩阵`B`


```python
B = np.outer(A.sum(1), A.sum(0)) / A.sum()
```

利用卡方值公式，计算相应取值


```python
res = ((A - B) ** 2 / B).sum()
res
```

**参考答案：**


```python
B = A.sum(0) * A.sum(1).reshape(-1, 1) / A.sum()
res = ((A - B) ** 2 / B).sum()
res
```

**总结：**

我的解答根据公式计算，可理解为外连接构造矩阵`B`，但是参考答案解题思路更加直观，便于理解。

### 2.4 Ex4：改进矩阵计算的性能
设$Z$为$m×n$的矩阵，$B$和$U$分别是$m×p$和$p×n$的矩阵，$B_i$为$B$的第$i$行，$U_j$为$U$的第$j$列，下面定义$\displaystyle R=\sum_{i=1}^m\sum_{j=1}^n\|B_i-U_j\|_2^2Z_{ij}$，其中$\|\mathbf{a}\|_2^2$表示向量$a$的分量平方和$\sum_i a_i^2$。

现有某人根据如下给定的样例数据计算$R$的值，请充分利用`Numpy`中的函数，基于此问题改进这段代码的性能。


```python
np.random.seed(0)
m, n, p = 100, 80, 50
B = np.random.randint(0, 2, (m, p))
U = np.random.randint(0, 2, (p, n))
Z = np.random.randint(0, 2, (m, n))
def solution(B=B, U=U, Z=Z):
    L_res = []
    for i in range(m):
        for j in range(n):
            norm_value = ((B[i]-U[:,j])**2).sum()
            L_res.append(norm_value*Z[i][j])
    return sum(L_res)
solution(B, U, Z)
```




    100566



**我的解答：**  
根据题意，可知道是$B$的第$i$行$-U$的第$j$行，属于两个长度为50的向量相减，然后根据分量平方和，得到$(i,j)$所在位置的值


```python
B.shape,U.T.shape
```




    ((100, 50), (80, 50))



观察上面，可以B的每一行，都需要与U的转置矩阵的每一行相减，然后计算分量平方和，可通过下面试探方式进行代码编写。


```python
B0 = ((B[0] - U.T) ** 2).sum(1)
B1 = ((B[1] - U.T) ** 2).sum(1)
```


```python
np.c_[B0, B1].shape
```




    (80, 2)



根据上述代码，可知遍历$m$，然后将矩阵拼接起来，得到$\|B_i-U_j\|_2^2$计算出来的矩阵，观察发现该矩阵维度是(80,100)


```python
def solution_improved(B=B, U=U, Z=Z):
    for i in range(m):
        if i==0:
            res = ((B[i] - U.T) ** 2).sum(1)
        else:
            res = np.c_[res, ((B[i] - U.T) ** 2).sum(1)]
    # 将矩阵转置再乘以矩阵Z，最后求和即可得到结果
    return (res.T * Z).sum()
```


```python
assert solution_improved(B, U, Z) == 100566
```


```python
%timeit -n 30 solution(B, U, Z)
```

    30 loops, best of 5: 78.9 ms per loop
    


```python
%timeit -n 30 solution_improved(B, U, Z)
```

    30 loops, best of 5: 6.77 ms per loop
    

**参考答案：**


```python
def solution_ref(B, U, Z):
    return (((B**2).sum(1).reshape(-1,1) + (U**2).sum(0) - 2*B@U)*Z).sum()
```


```python
%timeit -n 30 solution_ref(B, U, Z)
```

    30 loops, best of 5: 965 µs per loop
    

观察可知，我的解法比参考答案在时间复杂度上高了10倍，原因在于我使用了一个`for`循环，导致复杂度提升。

### 2.5 Ex5：连续整数的最大长度

输入一个整数的`Numpy`数组，返回其中递增连续整数子数组的最大长度，正向是指递增方向。例如，输入\[1,2,5,6,7\]，\[5,6,7\]为具有最大长度的连续整数子数组，因此输出3；输入\[3,2,1,2,3,4,6\]，\[1,2,3,4\]为具有最大长度的连续整数子数组，因此输出4。请充分利用`Numpy`的内置函数完成。（提示：考虑使用`nonzero, diff`函数）

**我的解答：**  
根据提示，可知采用`diff`函数，可以将此题转化为“一个数组内连续1的子数组的最大长度”，可以使用`map`和`join`来解决此问题
1. 使用`diff`函数，将数组转换为相邻元素的差值为1，即为连续的整数
2. 使用`where`将所有非1的元素都转换为0
3. 使用`splits`函数在0处分割将数组转换成字符串
4. 再获取子串的最大长度就是最大连续`1`的长度
5. 因为采用了diff函数，数组长度会缩短1，故加上1即可


```python
def count_len(A):
    res = np.diff(A)
    res = np.where(res==1, res, 0)
    return max(map(len, ''.join(map(str, res)).split('0'))) + 1
```


```python
A = np.array([1,2,5,6,7])
res = count_len(A)
res
```




    3




```python
A = np.array([3,2,1,2,3,4,6])
res = count_len(A)
res
```




    4



**参考答案：**


```python
def count_len_ref(A):
    return np.diff(np.nonzero(np.r_[1,np.diff(A)!=1,1])).max()
```

比较两个解答的时间复杂度


```python
A = np.array([6,4,3,2,1,2,3,4,5,6,7,10,9,4,6])
```


```python
%timeit -n 3000 count_len(A)
```

    3000 loops, best of 5: 34.8 µs per loop
    


```python
%timeit -n 3000 count_len_ref(A)
```

    3000 loops, best of 5: 73.2 µs per loop
    

对比出来的结果差不多，时间复杂度都是$O(n)$
