# Task02 怎样寻找最好的算法

## 1 总和最大区间问题

- 例题（股票最长有效增长期问题）：给定一个实数序列，设计一个最有效的算法，找到一个总和最大的区间
- 可行方法：
    1. 三重循环：两重循环进行遍历$p \leftarrow (1, K)$，$q \leftarrow (p, K)$，在每种组合中，计算$S(p,q)$平均要做$K/4$次加法，复杂度$O(K^3)$
    2. 两重循环：进行两次循环，定位$p,q$两个值，然后记录三个中间值，分别是从$p$开始到当前位置$q$的总和$S(p,q)$、从$p$到$q$为止总和的最大值、区间结束的位置，复杂度$O(K^2)$
    3. 分治算法：将序列分为$[1,K/2]$和$[K/2+1,K]$，计算两种情况：前后两个子序列的总和最大区间中间没有间隔（大中取大）、前后两个子序列的总和最大区间中间有间隔（$\max([p_1, q_1],[p_2,q_2],[p_1, q_2])$），复杂度$O(K\log K)$
    4. 正、反两遍扫描：在序列中找到第一个大于0的数$p$，计算$S(p,q)$和最大值$Maxf$和到达最大值的右边界$r$；利用同样的方法，反向扫描得到左边界$l$，则整个总和最大的区间是$[l,s]$，复杂度$O(K)$
- 对第4个方法的优化：
    - 确定$S(p,q)$总是大于0的边界$q$，然后反向计算$Maxb$，可得到$[l_1,r_1]$，$S(l_1, r_1)=Max_1$
    - 从$q+1$向后扫描，得到局部和最大区域$[l_2, r_2]$，$S(l_2, r_2)=Max_2$
    - 比较$Max_1$、$Max_2$和$Max_1 + Max_2 + S(r_1 + 1, l_2 - 1)$中最大值
    - 向后扫描，将局部和进行比较，更新$Max$

## 2 计算机算法思维感觉的建立

- 目标：将计算机算法灵活应用，领悟计算机科学的精妙之处
- 体会：
    1. 对一个问题边界的认识
    2. 优化算法常用的方法：检查一种算法是否在做大量无用功
    3. 逆向思维：从后往前思考问题

## 3 课后思考题

### 3.1 Q1：将例题1.3的线性复杂度算法写成伪代码

**解答：**
```python
def maxSubArray(nums: list) -> (int, list):
    # 区间和
    sub_sum = 0
    # 整个区间的最大值
    max_sum = 0
    # 左边界
    left = 0
    # 右边界
    right = 0
    # 第一个大于零的位置
    p = 0
    # (1) 先在序列中扫描找到第一个大于零的数
    for i in range(len(nums)):
        if nums[i] > 0:
            p = i
            break

    for q in range(p, len(nums)):
        # 计算从0~i的和
        sub_sum += nums[q]
        # （4）比较局部最大和
        if sub_sum > max_sum:
            # 更新max
            max_sum = sub_sum
            # 记录左右边界
            right = q
            left = p
        # （2）当s<0时
        if sub_sum < 0:
            sub_sum = 0
            # （3）从q+1开始往后扫描
            p = q + 1

    return max_sum, nums[left:right + 1]
```

### 3.2 Q2：在一个数组中寻找一个区间，使得区间内的数字之和等于某个事先给定的数字。

**解答：**
本题和两数之和比较类似，可采用相同的思路。
```python
def target_sub_array(target: int, nums: list):
    # 定义哈希表{key=sub_sum_value, value=right}
    hash_dict = {}
    # 区间和
    sub_sum = 0
    for q in range(len(nums)):
        sub_sum += nums[q]

        # 情况1：S(1, q) == target
        if sub_sum == target:
            return nums[0:q + 1]

        # 情况2：S(p, q) == target
        # S(p, q) = S(1, q) - S(1, p - 1)
        # S(1, p - 1) = S(1, q) - target
        if sub_sum - target in hash_dict.keys():
            p = hash_dict[sub_sum - target]
            return nums[p + 1:q + 1]

        if sub_sum not in hash_dict.keys():
            hash_dict[sub_sum] = q
```

### 3.3 Q3：在一个二维矩阵中，寻找一个矩形的区域，使其中的数字之和达到最大值。

**解答：**
```python
def get_max_matrix(matrix):
    n = len(matrix)
    m = len(matrix[0])
    b = [0] * m
    max_sum = -float('inf')
    best_r1, best_c1 = 0, 0
    r1, c1, r2, c2 = 0, 0, 0, 0

    for i in range(n):
        # 变更子矩阵的时候，都将b数组清0
        for t in range(m):
            b[t] = 0

        for j in range(i, n):
            sub_sum = 0
            for k in range(m):
                b[k] += matrix[j][k]

                # 计算最大子数组
                if sub_sum > 0:
                    sub_sum += b[k]
                else:
                    sub_sum = b[k]
                    best_r1 = i
                    best_c1 = k

                if sub_sum > max_sum:
                    max_sum = sub_sum
                    # 更新值
                    r1 = best_r1
                    c1 = best_c1
                    r2 = j
                    c2 = k

    return max_sum, [r1, c1, r2, c2]
```