#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: my_groupby.py
@time: 2020/12/26 0:27
@project: my-team-learning
@desc: transform函数的实现
"""

import numpy as np
import pandas as pd


class my_groupby:
    def __init__(self, df: pd.DataFrame, group_cols):
        # 原始数据集，进行复制
        self._df = df.copy()
        # 得到分组类别，返回Series类型
        self._groups = self._df[group_cols].drop_duplicates()
        # 如果取出为Series，需要转换为DataFrame
        if isinstance(self._groups, pd.Series):
            self._group_category_df = self._groups.to_frame()
        else:
            self._group_category_df = self._groups.copy()

    def __getitem__(self, col):
        # 由于要满足[col]/[[col1, col2, ...]]，故需要getitem方法
        # 为保证head()方法的使用，需要返回DataFrame或Series类型
        self._pr_col = [col] if isinstance(col, str) else list(col)
        return self

    def transform(self, my_func) -> pd.Series:
        # 定义两个空数组，用于存储索引和数据
        index_array = np.array([])
        value_array = np.array([])
        for group_df in self.__iter_group():
            # 进行分组遍历
            if self._pr_col:
                group_df = group_df[self._pr_col]
            if group_df.shape[1] == 1:
                group_df = group_df.iloc[:, 0]
            # 执行自定义函数
            group_res = my_func(group_df)
            # 转换为Series，用于进行数据拼接
            if not isinstance(group_res, pd.Series):
                group_res = pd.Series(group_res, index=group_df.index,
                                      name=group_df.name)

            # 存储索引和数据
            index_array = np.r_[index_array, group_res.index]
            value_array = np.r_[value_array, group_res.values]

        # 将分组之后得到的数据，再进行重排，按照正常索引之后的数据
        values = pd.Series(data=value_array, index=index_array).sort_index().values
        # 结合原始数据集的索引，构建Series
        result_series = pd.Series(data=values, index=self._df.reset_index().index, name=my_func.__name__)
        return result_series

    def head(self, n=5) -> pd.DataFrame:
        '''
        该功能是每个分组取前n个
        :param n:
        :return:
        '''
        # 取每个分组的前n个数据
        res_df = pd.DataFrame()
        for group_df in self.__iter_group():
            res_df = pd.concat([res_df, group_df.head(n)], ignore_index=True)

        # 删除索引列，重新指定索引数据
        index_values = res_df['index'].values
        res_df = res_df.drop(axis=0, columns=['index'])
        res_df.index = index_values

        if self._pr_col:
            return res_df[self._pr_col]
        return res_df

    def __iter_group(self):
        # 进行分组
        for index, groups in self._group_category_df.iterrows():
            group_df = self._df.reset_index().copy()
            for col_name, target in groups.to_dict().items():
                group_df = group_df[group_df[col_name] == target]
            yield group_df


if __name__ == '__main__':
    df = pd.read_csv('../data/car.csv')

    print("原始数据集：\n", df.head())


    def test1():
        # head()方法测试
        res = my_groupby(df, ['Type', 'Country'])['Disp.', 'HP'].head(1)
        res_df = df.groupby(['Type', 'Country'])[['Disp.', 'HP']].head(1)
        assert res.equals(res_df)


    def f(s):
        res = (s - s.min()) / (s.max() - s.min())
        return res


    def test2():
        # 单列分组
        res = my_groupby(df, 'Type')['Price'].transform(f).head()
        res_df = df.groupby('Type')['Price'].transform(f).head()
        assert res.equals(res_df)


    def test3():
        # 多列分组
        res = my_groupby(df, ['Type', 'Country'])['Price'].transform(f).head()
        res_df = df.groupby(['Type', 'Country'])['Price'].transform(f).head()
        assert res.equals(res_df)


    def test4():
        # 标量广播
        res = my_groupby(df, 'Type')['Price'].transform(lambda x: x.mean()).head()
        res_df = df.groupby('Type')['Price'].transform(lambda x: x.mean()).head()
        assert res.equals(res_df)


    def test5():
        # 跨列计算
        res = my_groupby(df, 'Type')[['Disp.', 'HP']].transform(lambda x: x['Disp.'] / x.HP).head()
        print("\n跨列计算：\n", res)


    test1()
    test2()
    test3()
    test4()
    test5()
