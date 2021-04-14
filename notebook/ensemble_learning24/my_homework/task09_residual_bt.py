#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: task09_residual_bt.py
@time: 2021/4/14 20:37
@project: my-team-learning
@desc:
"""
import numpy as np


class Tree:
    def __init__(self, split_point, mse, left_value, right_value, residual):
        # feature最佳切分点
        self.best_split_point = split_point
        # 平方误差
        self.mse = mse
        # 左子树值
        self.left_value = left_value
        # 右子树值
        self.right_value = right_value
        # 每棵决策树生成后的残差
        self.residual = residual


class ResidualBT:
    def __init__(self, X, y, tol=0.05, n_estimators=6):
        # 训练数据：实例
        self.X = X
        # 训练数据：标签
        self.y = y
        # 最大迭代次数
        self.n_estimators = n_estimators
        # 回归树
        self.T = []

    def fit(self):
        """
        对训练数据进行学习
        :return:
        """

        # 得到切分点
        split_point = self.split_point()

        residual = self.y.copy()
        for i in range(self.n_estimators):
            tree, residual = self.build_desicion_tree(split_point, residual)
            self.T.append(tree)

    def predict(self, X):
        """
        对新数据进行预测
        """
        m = np.shape(X)[0]
        y_predict = np.zeros(m)

        for tree in self.T:
            for i in range(m):
                if X[i] < tree.best_split_point:
                    y_predict[i] += tree.left_value
                else:
                    y_predict[i] += tree.right_value
        return y_predict

    def sse(self):
        """平方损失误差"""
        y_predict = self.predict(X)
        return np.sum((y_predict - y) ** 2)

    def score(self, X, y):
        """对训练效果进行评价"""
        y_predict = self.predict(X)
        error_rate = np.sum(np.abs(y_predict - y)) / 2 / y.shape[0]
        return 1 - error_rate

    def split_point(self):
        """
        获取切分点
        :return: 切分点
        """
        return (self.X[0:-1] + self.X[1:]) / 2

    def build_desicion_tree(self, split_point, label):
        m_s_list = []
        c1_list = []
        c2_list = []
        for p in split_point:
            # 切分点左边的label值
            label_left = label[0:int(p)]
            # 切分点右边的label值
            label_right = label[int(p):]
            c1 = np.mean(label_left)
            c2 = np.mean(label_right)
            m_s = np.sum((label_left - c1) ** 2) + np.sum((label_right - c2) ** 2)
            c1_list.append(c1)
            c2_list.append(c2)
            m_s_list.append(m_s)
        # 得到m_s最小值所在的位置
        best_index = np.argmin(m_s_list)
        # 得到最优切分点
        best_split_point = split_point[int(best_index)]
        # 得到最优均方误差
        best_mse = m_s_list[int(best_index)]
        # 得到左子树的label值
        best_y_lf = label[0:int(best_split_point)]
        lf_value = np.mean(best_y_lf)
        # 得到右子树的label值
        best_y_rt = label[int(best_split_point):]
        rt_value = np.mean(best_y_rt)
        # 得到决策树的残差
        residual = np.concatenate((best_y_lf - lf_value, best_y_rt - rt_value,))
        tree = Tree(best_split_point, best_mse, lf_value, rt_value, residual)
        return tree, residual


if __name__ == '__main__':
    # 加载训练数据，数据集来自与教程的例题
    X = np.arange(1, 11)
    y = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])
    clf = ResidualBT(X, y, n_estimators=6)
    clf.fit()
    y_predict = clf.predict(X)
    score = clf.score(X, y)
    print("\n原始输出:", y)
    print("预测输出:", y_predict)
    print("预测正确率：{:.2%}".format(score))
    print("平方损失误差：{:.2}".format(clf.sse()))
