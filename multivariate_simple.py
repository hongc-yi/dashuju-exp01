"""
实验一：多元线性回归 — 花萼长度与花萼宽度、花瓣长度、花瓣宽度
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
iris = load_iris()
data = iris.data
feature_names = ['花萼宽度', '花瓣长度', '花瓣宽度']

# 自变量：花萼宽度(1)、花瓣长度(2)、花瓣宽度(3)
X = data[:, [1, 2, 3]]
y = data[:, 0]  # 因变量：花萼长度

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*50)
print("多元线性回归分析")
print("="*50)
print(f"自变量: {feature_names}")
print(f"因变量: 花萼长度")
print(f"样本数: {len(y)}")

# 2. 创建并训练模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 3. 模型参数
b = model.intercept_
w = model.coef_
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\n回归方程:")
print(f"花萼长度 = {b:.4f}", end="")
for i, name in enumerate(feature_names):
    sign = "+" if w[i] >= 0 else "-"
    print(f" {sign} {abs(w[i]):.4f} * {name}", end="")
print()

print(f"\n模型评估:")
print(f"  R² 决定系数: {r2:.4f}")
print(f"  R² (百分比): {r2*100:.2f}%")
print(f"  均方误差 MSE: {mse:.4f}")

# 4. 可视化：三个子图，每个特征一张（与一元风格一致）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (ax, name) in enumerate(zip(axes, feature_names)):
    # 实际数据点
    ax.scatter(X[:, i], y, c='steelblue', s=40, alpha=0.7, label='实际数据')
    # 多元模型预测值
    ax.scatter(X[:, i], y_pred, c='red', s=20, alpha=0.5, label='多元预测值')
    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel('花萼长度 (cm)', fontsize=11)
    ax.set_title(f'{name} vs 花萼长度', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'多元线性回归 — 各特征与花萼长度 (R²={r2:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 5. 与一元回归对比
print(f"\n{'='*50}")
print("与一元回归结果对比")
print(f"{'='*50}")
print(f"{'模型':<20} {'R²':<10} {'MSE':<10}")
print('-'*40)
print(f"{'多元回归 (全部特征)':<20} {r2:<10.4f} {mse:<10.4f}")

for i, name in enumerate(feature_names):
    X_single = X[:, i].reshape(-1, 1)
    m = LinearRegression().fit(X_single, y)
    r2_s = r2_score(y, m.predict(X_single))
    mse_s = mean_squared_error(y, m.predict(X_single))
    print(f"{'一元 (' + name + ')':<20} {r2_s:<10.4f} {mse_s:<10.4f}")

print(f"\n结论：")
print(f"1. 多元回归 R²={r2:.4f} 高于任意一元回归，说明多特征共同预测效果更好")
print(f"2. 多元回归解释了花萼长度 {r2*100:.2f}% 的方差")