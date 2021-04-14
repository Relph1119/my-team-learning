#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task09_adaboost.py
@time: 2021/4/14 20:37
@project: my-team-learning
@desc:
"""
import numpy as np


class AdaBoost:
    def __init__(self, X, y, tol=0.05, max_iter=10):
        # 训练数据 实例
        self.X = X
        # 训练数据 标签
        self.y = y
        # 训练中止条件 right_rate>self.tol
        self.tol = tol
        # 最大迭代次数
        self.max_iter = max_iter
        # 初始化样本权重w
        self.w = np.full((X.shape[0]), 1 / X.shape[0])
        # 弱分类器
        self.G = []

    def build_stump(self):
        """
        以带权重的分类误差最小为目标，选择最佳分类阈值
        best_stump['dim'] 合适的特征所在维度
        best_stump['thresh'] 合适特征的阈值
        """
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)
        m, n = np.shape(self.X)
        # 分类误差
        e_min = np.inf
        # 小于分类阈值的样本属于的标签类别
        sign = None
        # 最优分类树桩
        best_stump = {}
        for i in range(n):
            # 求每一种特征的最大最小值
            range_min = self.X[:, i].min()
            range_max = self.X[:, i].max()
            for thresh_val in np.arange(range_min, range_max, 0.5):
                for intq in ['lt', 'lf']:
                    predict_vals = self.base_estimator(self.X, i, thresh_val, intq)
                    err_arr = np.ones(m)
                    err_arr[predict_vals.T == self.y.T] = 0
                    weighted_error = np.dot(self.w, err_arr)
                    if weighted_error < e_min:
                        e_min = weighted_error
                        sign = predict_vals
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['intq'] = intq
        return best_stump, sign, e_min

    def updata_w(self, alpha, predict):
        """
        更新样本权重w
        """
        # 以下2行根据公式更新样本权重
        P = self.w * np.exp(-alpha * self.y * predict)
        self.w = P / P.sum()

    @staticmethod
    def base_estimator(X, dimen, threshVal, intq):
        """
        计算单个弱分类器预测输出
        """
        # 预测矩阵
        ret_array = np.ones(np.shape(X)[0])
        if intq == 'lt':
            if len(X.shape) == 1:
                ret_array[X >= threshVal] = -1.0
            else:
                ret_array[X[:, dimen] >= threshVal] = -1.0
        else:
            if len(X.shape) == 1:
                ret_array[X < threshVal] = -1.0
            else:
                ret_array[X[:, dimen] < threshVal] = -1.0
        return ret_array

    def fit(self):
        """
        对训练数据进行学习
        """
        G = 0
        for i in range(self.max_iter):
            # 获取当前迭代最佳分类阈值
            best_stump, sign, error = self.build_stump()
            # 计算本轮弱分类器的系数
            alpha = 1 / 2 * np.log((1 - error) / error)
            # 弱分类器权重
            best_stump['alpha'] = alpha
            # 保存弱分类器
            self.G.append(best_stump)
            # 以下3行计算当前总分类器（之前所有弱分类器加权和）分类效率
            G += alpha * sign
            y_predict = np.sign(G)
            error_rate = np.sum(np.abs(y_predict - self.y)) / 2 / self.y.shape[0]
            print("第%i次迭代" % (i + 1))
            self.print_fun()

            if error_rate < self.tol:
                # 满足中止条件 则跳出循环
                break
            else:
                # 若不满足，更新权重，继续迭代
                self.updata_w(alpha, y_predict)

    def predict(self, X):
        """
        对新数据进行预测
        """
        m = np.shape(X)[0]
        G = np.zeros(m)
        for i in range(len(self.G)):
            stump = self.G[i]
            # 遍历每一个弱分类器，进行加权
            _G = self.base_estimator(X, stump['dim'], stump['thresh'], stump['intq'])
            alpha = stump['alpha']
            G += alpha * _G
        y_predict = np.sign(G)
        return y_predict.astype(int)

    def score(self, X, y):
        """对训练效果进行评价"""
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
        return 1 - error_rate

    def print_fun(self):
        i = 1
        s = "f(x)="
        for stump in self.G:
            if i != 1:
                s += "+"
            s += "{}·G{}(x)".format(round(stump['alpha'], 4), i)
            i += 1
        print(s)


if __name__ == '__main__':
    # 加载训练数据，数据集来自与教程的例题
    X = np.arange(10)
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    clf = AdaBoost(X, y, max_iter=10)
    clf.fit()
    y_predict = clf.predict(X)
    score = clf.score(X, y)
    print("\n原始输出:", y)
    print("预测输出:", y_predict)
    print("预测正确率：{:.2%}".format(score))
