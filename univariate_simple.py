"""
实验一：一元线性回归 — IRIS花萼长度与各特征的一元关系
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
iris = load_iris()
data = iris.data
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']

# 因变量：花萼长度 (索引0)
y = data[:, 0]

# 三个自变量：花萼宽度(1)、花瓣长度(2)、花瓣宽度(3)
x_indices = [1, 2, 3]
x_names = [feature_names[i] for i in x_indices]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 2. 分别进行一元线性回归
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (x_idx, x_name) in enumerate(zip(x_indices, x_names)):
    X = data[:, x_idx].reshape(-1, 1)

    # 创建并训练模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 模型参数
    w = model.coef_[0]
    b = model.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print(f"\n{'='*50}")
    print(f"一元回归：花萼长度 vs {x_name}")
    print(f"{'='*50}")
    print(f"回归方程: y = {b:.4f} + {w:.4f} * x")
    print(f"R² 决定系数: {r2:.4f}")
    print(f"均方误差 MSE: {mse:.4f}")

    # 散点 + 拟合直线
    ax = axes[i]
    ax.scatter(X, y, c='steelblue', s=40, alpha=0.7, label='数据点')
    X_line = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, 'r-', linewidth=2, label=f'y={b:.2f}+{w:.2f}x')
    ax.set_xlabel(x_name, fontsize=11)
    ax.set_ylabel('花萼长度 (cm)', fontsize=11)
    ax.set_title(f'{x_name} vs 花萼长度\nR²={r2:.4f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('一元线性回归分析 — IRIS数据集', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. 结果汇总
print(f"\n{'='*50}")
print("结果汇总")
print(f"{'='*50}")
print(f"{'自变量':<12} {'回归方程':<30} {'R²':<8} {'MSE':<8}")
print('-'*58)
for x_idx, x_name in zip(x_indices, x_names):
    X = data[:, x_idx].reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    w, b = model.coef_[0], model.intercept_
    r2 = r2_score(y, model.predict(X))
    mse = mean_squared_error(y, model.predict(X))
    print(f"{x_name:<12} y={b:.2f}+{w:.2f}x{'':<12} {r2:<8.4f} {mse:<8.4f}")

print(f"\n结论：花瓣长度与花萼长度的线性关系最强 (R²最高)")