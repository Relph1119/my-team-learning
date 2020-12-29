#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task06_ex2_join.py
@time: 2020/12/29 22:36
@project: my-team-learning
@desc: 实现带有how参数的join函数
"""

import numpy as np
import pandas as pd


def join(df1: pd.DataFrame, df2: pd.DataFrame, how='left') -> pd.DataFrame:
    # 得到所有要连接的列
    res_col = df1.columns.tolist() + df2.columns.tolist()
    # 得到无重复的索引
    index_dup = df1.index.unique().intersection(df2.index.unique())
    # 建立空的DataFrame用于连接
    res_df = pd.DataFrame(columns=res_col)

    # 构造笛卡尔积的DataFrame
    for label in index_dup:
        # 防止单个str字符串被list()函数拆成单个字符列表
        df1_values = df1.loc[label].values.reshape(-1, 1)
        df2_values = df2.loc[label].values.reshape(-1, 1)

        # 计算笛卡尔积
        cartesian = [list(i) + list(j) for i in df1_values for j in df2_values]
        # 构造笛卡尔积的DataFrame
        res_df = create_dateframe(cartesian, label, res_col, res_df)

    if how in ['left', 'outer']:
        # 遍历df1，进行连接
        for label in df1.index.unique().difference(index_dup):
            df_lable = validate_dataframe(df1, label)
            cat = [list(i) + [np.nan] * df2.shape[1] for i in df_lable.values]
            # 构建DataFrame
            res_df = create_dateframe(cat, label, res_col, res_df)
    if how in ['right', 'outer']:
        # 遍历df2，进行连接
        for label in df2.index.unique().difference(index_dup):
            df_lable = validate_dataframe(df2, label)
            cat = [[np.nan] * df1.shape[1] + list(i) for i in df_lable.values]
            # 构建DataFrame
            res_df = create_dateframe(cat, label, res_col, res_df)

    res_df = columns_dtype_convert(res_df, df1, df2)
    return res_df


def validate_dataframe(df: pd.DataFrame, label) -> pd.DataFrame:
    '''
    判断df.loc[lable]是否为DataFrame，如果不是将其转换为DataFrame
    :param df:
    :param label:
    :return:
    '''
    df_lable = df.loc[label]
    if not isinstance(df_lable, pd.DataFrame):
        df_lable = df.loc[label].to_frame()
    return df_lable


def create_dateframe(data, label, res_col, res_df: pd.DataFrame) -> pd.DataFrame:
    dup_df = pd.DataFrame(data, index=[label] * len(data), columns=res_col)
    res_df = pd.concat([res_df, dup_df])
    return res_df


def columns_dtype_convert(res_df: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # 进行类型转换
    df1_dtypes = dict(df1.dtypes)
    df2_dtypes = dict(df2.dtypes)
    df1_dtypes.update(df2_dtypes)

    res_number_cols = df1.select_dtypes('number').columns.tolist() + df2.select_dtypes('number').columns.tolist()
    # 只留下数据里面有nan的列
    nan_cols = []
    for col in res_number_cols:
        if res_df[col].isna().any():
            nan_cols.append(col)
    # 将number类型的数据转换为float64类型数据
    res_df = res_df.astype(dict(zip(nan_cols, ['float64'] * len(nan_cols))))
    # 在转换列中删除数据为nan的列
    if nan_cols:
        df1_dtypes.pop(*nan_cols)
    res_df = res_df.astype(df1_dtypes)

    return res_df


if __name__ == '__main__':
    def test(df1, df2, how):
        df = df1.join(df2, how=how)
        res = join(df1, df2, how=how)
        assert res.equals(df)


    df1 = pd.DataFrame(columns=['Name'], data=['LiSi', 'ZhangSan', 'WangWu', 'XiaoLiu'],
                       index=['A', 'B', 'B', 'C'])
    df2 = pd.DataFrame(columns=['Grade'], data=[92, 64, 73, 80],
                       index=['A', 'A', 'B', 'D'])

    test(df1, df2, 'left')
    test(df1, df2, 'right')
    test(df1, df2, 'inner')
    test(df1, df2, 'outer')
