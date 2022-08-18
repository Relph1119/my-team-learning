# Task04 更好的排序算法

## 1 蒂姆排序（Timsort）

- 来源：由蒂姆·彼得斯（Tim Peters）将插入排序和归并排序的特点相结合，将最坏时间复杂度控制在$O(N \log N)$，同时保证排序稳定性
- 特点：以块为单位的归并排序，这些块的内部元素都是排好序的
- 算法步骤：
1. 找出序列中各个递增和递减的子序列：用插入排序整理成有序的子序列，并将子序列放入临时存储空间中
2. 按照规则合并这些块：按照块的长短进行合并（先合并两个最短的），采用跳跃式的方式进行成组归并

## 2 为什么排序算法的复杂度不可能小于O(NlogN)

### 2.1 重新定义排序
- 序列大小：有两个序列$a_1, a_2, \cdots, a_N$和$b_1, b_2, \cdots, b_N$，如果$a_i \leqslant b_i$，且$a_1 = b_1, a_2 = b_2, \cdots, a_{i - 1} = b_{i - 1}$，则称第1个序列小于第2个序列
- 重新定义排序：对于任意一个序列$a_1, a_2, \cdots, a_N$，最小的序列就是将每个元素从小到大排好序的序列

### 2.2 排序算法复杂度推导

$\because$ 如果进行$k$次序列比较，可能区分出$2^k$种不同的序列  
$\therefore$ 有$M$种序列，需要进行$\log M$次比较才能区分序列的大小  
$\because$ $N$个元素组成的数组可以排列出$N!$种序列  
$\therefore$ 选出最小的序列，需要$\log N!$次比较  
$\because$ 根据斯特林公式：$\ln N! = N \ln N - N + O(\ln N)$  
$\therefore$ $\log N! = O(N \log N)$  
故，任何排序算法的复杂度都不会低于$O(N \log N)$

## 3 课后思考题

### 3.1 Q1：赛跑问题（GS）

&emsp;&emsp;假定有25名短跑选手比赛争夺前三名，赛场上有五条赛道，一次可以有五名选手同时比赛。比赛并不计时，只看相应的名次。假设选手的发挥是稳定的，也就是说如果约翰比张三跑得快，张三比凯利跑得快，约翰一定比凯利跑得快。最少需要几次比赛才能决出前三名？

**解答：**

```python
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
```
执行结果如下：
```
The speed of all players are [23, 13, 19, 6, 7, 6, 7, 6, 22, 22, 3, 23, 25, 5, 23, 10, 24, 25, 1, 15, 15, 24, 21, 4, 1]
The number of track is 5
The minimum of counts is 7 times
The speed of Top3 is [1, 1, 3]
The number of Top3 players is [18, 24, 10]
```

### 3.2 Q2：区间排序

&emsp;&emsp;如果有$N$个区间$[l_1, r_1] , [l_2, r_2], \cdots, [l_N,r_N]$，只要满足下面的条件，我们就说这些区间是有序的：存在$x_i \in [l_i,r_i]$，其中$i = 1, 2, \cdots, N$。

&emsp;&emsp;比如，$[1, 4]$、$[2, 3]$和$[1.5, 2.5]$是有序的，因为我们可以从这三个区间中选择1.1、2.1和2.2三个数。同时$[2, 3]$、$[1, 4]$和$[1.5, 2.5]$也是有序的，因为我们可以选择2.1、2.2和2.4。但是$[1, 2]$、$[2.7, 3.5]$和$[1.5, 2.5]$不是有序的。

&emsp;&emsp;对于任意一组区间，如何将它们进行排序？

**解答：**
