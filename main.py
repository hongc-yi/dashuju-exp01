"""
实验一：一元线性回归算法Python实现（无需外部文件版本）
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


class LinearRegressionGD:
    """使用梯度下降法实现的一元线性回归模型"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta0 = 0
        self.theta1 = 0
        self.cost_history = []

    def compute_cost(self, X, y):
        m = len(y)
        predictions = self.theta0 + self.theta1 * X
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        m = len(y)
        self.cost_history = []

        for i in range(self.n_iterations):
            predictions = self.theta0 + self.theta1 * X
            gradient_theta0 = (1 / m) * np.sum(predictions - y)
            gradient_theta1 = (1 / m) * np.sum((predictions - y) * X)
            self.theta0 -= self.learning_rate * gradient_theta0
            self.theta1 -= self.learning_rate * gradient_theta1
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            if i % 200 == 0:
                print(f"迭代 {i}, 损失: {cost:.6f}")

    def predict(self, X):
        return self.theta0 + self.theta1 * X

    def get_params(self):
        return self.theta0, self.theta1


def run_pizza_experiment():
    """实验1：披萨直径与价格的一元线性回归分析（使用内置数据）"""
    print("=" * 60)
    print("实验1：披萨直径与价格的一元线性回归分析")
    print("=" * 60)

    # 直接定义数据（不需要外部文件）
    data = {
        'Id': [1, 2, 3, 4, 5],
        'Diameter': [6, 8, 10, 14, 18],
        'Price': [7, 9, 13, 17.5, 18]
    }
    pizza = pd.DataFrame(data)
    pizza.set_index('Id', inplace=True)

    print("\n数据集：")
    print(pizza)

    diameter = pizza['Diameter'].values
    price = pizza['Price'].values

    print(f"\n直径数据: {diameter}")
    print(f"价格数据: {price}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 散点图
    axes[0, 0].scatter(diameter, price, color='darkblue', s=100)
    axes[0, 0].set_xlabel('直径 (英寸)')
    axes[0, 0].set_ylabel('价格 (美元)')
    axes[0, 0].set_title('披萨直径与价格的关系')
    axes[0, 0].grid(True, alpha=0.3)

    # 训练模型
    print("\n开始训练模型...")
    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(diameter, price)

    theta0, theta1 = model.get_params()
    print(f"\n训练完成！")
    print(f"模型参数：截距 = {theta0:.4f}, 斜率 = {theta1:.4f}")
    print(f"线性回归方程：价格 = {theta0:.4f} + {theta1:.4f} × 直径")

    # 拟合直线
    diameter_pred = np.array([0, 25])
    price_pred = model.predict(diameter_pred)

    axes[0, 1].scatter(diameter, price, color='darkblue', s=100, label='实际数据')
    axes[0, 1].plot(diameter_pred, price_pred, color='red', linewidth=2, label='拟合直线')
    axes[0, 1].set_xlabel('直径 (英寸)')
    axes[0, 1].set_ylabel('价格 (美元)')
    axes[0, 1].set_title('一元线性回归拟合结果')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 25)
    axes[0, 1].set_ylim(0, 25)

    # 残差图
    price_pred_train = model.predict(diameter)
    for x, y_true, y_pred in zip(diameter, price, price_pred_train):
        axes[1, 0].plot([x, x], [y_true, y_pred], 'gray', alpha=0.5)
    axes[1, 0].scatter(diameter, price, color='darkblue', s=100, label='实际值')
    axes[1, 0].scatter(diameter, price_pred_train, color='red', marker='x', s=100, label='预测值')
    axes[1, 0].set_xlabel('直径 (英寸)')
    axes[1, 0].set_ylabel('价格 (美元)')
    axes[1, 0].set_title('残差图')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 损失收敛曲线
    axes[1, 1].plot(range(len(model.cost_history)), model.cost_history, 'b-')
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('损失值')
    axes[1, 1].set_title('梯度下降损失收敛曲线')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 评估
    mse = np.mean((price_pred_train - price) ** 2)
    r2 = 1 - np.sum((price - price_pred_train) ** 2) / np.sum((price - np.mean(price)) ** 2)

    print("\n模型评估：")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 R²: {r2:.4f}")

    return model


def run_iris_univariate_experiment():
    """实验2：Iris数据集一元线性回归"""
    print("\n" + "=" * 60)
    print("实验2：Iris数据集 - 花萼长度 vs 花萼宽度")
    print("=" * 60)

    iris = load_iris()
    X = iris.data[:, 0]  # 花萼长度
    y = iris.data[:, 1]  # 花萼宽度

    print(f"\n样本数量: {len(X)}")
    print(f"花萼长度范围: {X.min():.2f} ~ {X.max():.2f} cm")
    print(f"花萼宽度范围: {y.min():.2f} ~ {y.max():.2f} cm")

    # 标准化
    X_mean, X_std = np.mean(X), np.std(X)
    X_stdized = (X - X_mean) / X_std

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(X, y, alpha=0.7, c='steelblue', s=60)
    axes[0, 0].set_xlabel('花萼长度 (cm)')
    axes[0, 0].set_ylabel('花萼宽度 (cm)')
    axes[0, 0].set_title('花萼长度 vs 花萼宽度')
    axes[0, 0].grid(True, alpha=0.3)

    print("\n开始训练...")
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=2000)
    model.fit(X_stdized, y)

    theta0, theta1 = model.get_params()
    theta0_orig = theta0 - theta1 * X_mean / X_std
    theta1_orig = theta1 / X_std

    print(f"\n原始尺度模型：")
    print(f"花萼宽度 = {theta0_orig:.4f} + {theta1_orig:.4f} × 花萼长度")

    X_sorted_idx = np.argsort(X)
    X_sorted = X[X_sorted_idx]
    y_pred_sorted = model.predict((X_sorted - X_mean) / X_std)

    axes[0, 1].scatter(X, y, alpha=0.7, c='steelblue', s=60, label='实际数据')
    axes[0, 1].plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='拟合直线')
    axes[0, 1].set_xlabel('花萼长度 (cm)')
    axes[0, 1].set_ylabel('花萼宽度 (cm)')
    axes[0, 1].set_title('一元线性回归拟合结果')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(range(len(model.cost_history)), model.cost_history, 'g-')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('损失值')
    axes[1, 0].set_title('损失收敛曲线')
    axes[1, 0].grid(True, alpha=0.3)

    y_pred = model.predict((X - X_mean) / X_std)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)

    axes[1, 1].hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('残差分布')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n评估结果：")
    print(f"MSE: {mse:.6f}, R²: {r2:.6f}")

    if theta1_orig < 0:
        print(f"\n结论：花萼长度与花萼宽度呈负相关")


if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    run_pizza_experiment()
    run_iris_univariate_experiment()

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)