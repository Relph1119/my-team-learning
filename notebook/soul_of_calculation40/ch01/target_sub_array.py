#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: target_sub_array.py
@time: 2022/8/16 18:46
@project: my-team-learning
@desc: 习题1.3 Q2 目标值的子数组和
"""


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


if __name__ == '__main__':
    assert target_sub_array(6, [-1, -2, 1, -3, 4, -1, 2, 1, -5, 4]) == [4, -1, 2, 1]
    assert target_sub_array(1, [1]) == [1]
    assert target_sub_array(23, [5, 4, -1, 7, 8]) == [5, 4, -1, 7, 8]
