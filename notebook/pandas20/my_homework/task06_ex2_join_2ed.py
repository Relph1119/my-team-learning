import numpy as np
import pandas as pd


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


def my_join(df1, df2, how):
    # 将Series与Series或DataFrame进行合并
    def s_sord(s_a, b, index):
        # 利用.T倒置结果
        if isinstance(b, pd.Series):
            return pd.concat([s_a, b]).to_frame().T
        else:
            return pd.concat([pd.DataFrame([s_a] * b.shape[0]), b], axis=1)

    # 创建指定index和name的值为NaN的Series 需要判断长度
    def produceNanSer(index, name):
        return pd.Series([np.NaN] * index.shape[0], index=index, name=name)

    # 采用最开始第一节个人的想法，更换连接主体，递归调用自己的left方法

    if how == 'right':
        res_df = my_join(df2, df1, 'left')[list(df1.columns) + list(df2.columns)]
    # 另外三种情况：
    else:
        res = []
        # 遍历df1的每一行 根据index右表对应行数有三种情况 分别为0，1，n
        for i in range(df1.shape[0]):
            index = df1.index[i]
            # 1，n的情况
            if index in df2.index:
                res.append(s_sord(df1.iloc[i], df2.loc[index], index))
            # 0的情况
            else:
                # 此时how为inner时已经完成操作 不是则还须将左表独有的行加入
                if how != 'inner':
                    res.append(s_sord(df1.iloc[i], produceNanSer(df2.columns, index), index))
        # 此时how为left时已经完成操作 outer则还须将右表独有的行加入
        if how == 'outer':
            for i in range(df2.shape[0]):
                index = df2.index[i]
                if index not in df1.index:
                    res.append(s_sord(produceNanSer(df1.columns, index), df2.iloc[i], index))
        res_df = pd.concat(res).reset_index().sort_values('index').set_index('index')

    # 类型转换
    res_df = columns_dtype_convert(res_df, df1, df2)

    return res_df


if __name__ == '__main__':
    df1 = pd.DataFrame(columns=['Name'], data=['LiSi', 'ZhangSan', 'WangWu', 'XiaoLiu'],
                       index=['A', 'B', 'B', 'C'])
    df2 = pd.DataFrame(columns=['Grade'], data=[92, 64, 73, 80],
                       index=['A', 'A', 'B', 'D'])

    # df1 = pd.DataFrame({'col_1': [*'123456'],
    #                     'col_2': ['one', 'two', 'three', 'four', 'five', 'six']},
    #                    index=[*'abcbcd'])
    # df2 = pd.DataFrame({'col_3': [*'567890'],
    #                     'col_4': ['いち', 'に', 'さん', 'よん', 'ご', 'ろく']},
    #                    index=[*'dcecfe'])

    for x in ['left', 'right', 'inner', 'outer']:
        print(my_join(df1, df2, how=x).equals(df1.join(df2, how=x)))
        print('-------')
