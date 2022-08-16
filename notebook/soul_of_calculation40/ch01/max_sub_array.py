#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: max_sub_array.py
@time: 2022/8/16 16:59
@project: my-team-learning
@desc: 习题1.3 Q1 最大子数组和，算法复杂度O(K)
"""


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


if __name__ == '__main__':
    assert maxSubArray([-1, -2, 1, -3, 4, -1, 2, 1, -5, 4]) == (6, [4, -1, 2, 1])
    assert maxSubArray([1]) == (1, [1])
    assert maxSubArray([5, 4, -1, 7, 8]) == (23, [5, 4, -1, 7, 8])
