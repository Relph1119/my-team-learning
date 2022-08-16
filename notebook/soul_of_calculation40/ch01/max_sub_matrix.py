#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: max_sub_matrix.py
@time: 2022/8/16 19:36
@project: my-team-learning
@desc: 习题1.3 Q3 最大子矩阵和
"""


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


if __name__ == '__main__':
    matrix = [[-1, 0], [0, -1]]
    max_sum, ans = get_max_matrix(matrix)
    print(max_sum)
    print(ans)
