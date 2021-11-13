from CART import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.datasets import make_regression

if __name__ == "__main__":

    # 模拟回归数据集
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=0
    )

    my_cart = DecisionTreeRegressor(max_depth=2)
    my_cart.fit(X, y)
    res1 = my_cart.predict(X)
    importance1 = my_cart.feature_importances_

    sklearn_cart = dt(max_depth=2)
    sklearn_cart.fit(X, y)
    res2 = sklearn_cart.predict(X)
    importance2 = sklearn_cart.feature_importances_

    # 预测一致的比例
    print(((res1-res2)<1e-8).mean())
    # 特征重要性一致的比例
    print(((importance1-importance2)<1e-8).mean())