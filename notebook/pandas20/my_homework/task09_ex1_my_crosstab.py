#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task09_ex1_my_crosstab.py
@time: 2021/1/7 22:12
@project: my-team-learning
@desc: 实现带dropna参数的my_crosstab函数
"""

import numpy as np
import pandas as pd


def my_crosstab(s1: pd.Series, s2: pd.Series, dropna=True) -> pd.DataFrame:
    def validate_dropna(s: pd.Series):
        # 如果dropna=False并且series是分类变量
        if s.dtypes.name == 'category' and not dropna:
            return s.cat.categories
        # 否则返回series的不重复数据值作为行列索引
        return s.unique()

    s1_idx = validate_dropna(s1)
    s2_idx = validate_dropna(s2)
    # 构造全0数据
    data = np.zeros((s1_idx.shape[0], s2_idx.shape[0]))
    # 构造DataFrame，行索引为s1中的值，列索引为s2中的值
    res = pd.DataFrame(data, index=s1_idx, columns=s2_idx)
    res.rename_axis(index=s1.name, columns=s2.name, inplace=True)
    # 计算频数
    for s1_idx_value, s2_idx_value in zip(s1, s2):
        res.loc[s1_idx_value, s2_idx_value] += 1
    # 设置行索引
    res = res.astype(np.int64)
    return res


if __name__ == '__main__':
    df = pd.DataFrame({'A': ['a', 'b', 'c', 'a'], 'B': ['cat', 'cat', 'dog', 'cat']})


    def test1():
        res_my_crosstab_df = my_crosstab(df.A, df.B)
        res_crosstab_df = pd.crosstab(df.A, df.B)
        print(res_my_crosstab_df.equals(res_crosstab_df))


    def test2():
        df.B = df.B.astype('category').cat.add_categories('sheep')
        res_my_crosstab_df = my_crosstab(df.A, df.B, dropna=False)
        res_crosstab_df = pd.crosstab(df.A, df.B, dropna=False)
        print(res_my_crosstab_df.equals(res_crosstab_df))


    test1()
    test2()
