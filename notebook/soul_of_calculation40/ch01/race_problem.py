#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: race_problem.py
@time: 2022/8/16 22:06
@project: my-team-learning
@desc: 习题1.4 Q1：赛跑问题
"""
import random


def get_top3(nums: list, track_num):
    n = len(nums) // track_num
    groups = []
    count = 0
    # 将跑步速度与号码绑定(race_speed, number)
    nums_index = []
    for i in range(len(nums)):
        nums_index.append((nums[i], i))

    for i in range(n):
        count += 1
        if track_num * i + track_num < len(nums):
            groups.append(sorted(nums_index[track_num * i: track_num * i + track_num]))
        else:
            groups.append(sorted(nums_index[track_num * i:]))
    # 将每个队伍的第一名比赛一次(race_speed, number, group_no)
    no1_race = [(*g[0], i) for i, g in enumerate(groups)]
    no1_race.sort()
    count += 1

    # 取出前3个，然后加上后面两组的第2名
    N1 = no1_race[0]
    A2 = groups[no1_race[0][2]][1]
    B1 = no1_race[1]
    C1 = no1_race[2]
    B2 = groups[no1_race[1][2]][1]
    C2 = groups[no1_race[2][2]][1]
    no1_race = sorted([A2, B1, C1, B2, C2])
    count += 1

    N2, N3 = no1_race[0], no1_race[1]

    return count, [N1, N2, N3]


if __name__ == '__main__':
    random.seed(24)
    people_nums = 25
    nums = [random.randint(1, people_nums) for _ in range(people_nums)]
    print("The speed of all players are", nums)
    track_num = 5
    print("The number of track is", track_num)
    count, top3 = get_top3(nums, track_num)
    print("The minimum of counts is {} times".format(count))
    print("The speed of Top3 is", [i[0] for i in top3])
    print("The number of Top3 players is", [i[1] for i in top3])

    assert [i[0] for i in top3] == sorted(nums)[:3]
