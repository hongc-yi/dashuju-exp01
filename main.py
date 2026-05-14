"""
实验一：一元线性回归算法Python实现
数据集：IRIS（鸢尾花数据集）
分析花萼长度与花萼宽度、花瓣长度、花瓣宽度之间的关系
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ==================== 第一部分：自主实现线性回归 ====================

class LinearRegressionGD:
    """
    使用梯度下降法实现的线性回归模型
    支持一元和多元线性回归
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化线性回归模型

        参数:
            learning_rate: 学习率(步长)
            n_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None  # 参数向量，theta[0]是截距，theta[1:]是系数
        self.cost_history = []

    def compute_cost(self, X, y):
        """
        计算均方误差损失函数 J(theta) = (1/(2m)) * sum((h(x) - y)^2)
        """
        m = len(y)
        predictions = X @ self.theta[1:] + self.theta[0]
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        """
        使用批量梯度下降法训练模型

        参数:
            X: 特征值数组，形状 (m, n)，m为样本数，n为特征数
            y: 目标值数组，形状 (m,)
        """
        m, n = X.shape
        # 初始化参数
        self.theta = np.zeros(n + 1)
        self.cost_history = []

        for i in range(self.n_iterations):
            # 前向传播：计算预测值
            predictions = X @ self.theta[1:] + self.theta[0]

            # 计算梯度
            # theta0的梯度：d(J)/d(theta0) = (1/m) * sum(h(x) - y)
            grad0 = (1 / m) * np.sum(predictions - y)

            # theta1~thetan的梯度：d(J)/d(theta_j) = (1/m) * sum((h(x) - y) * x_j)
            grad_rest = (1 / m) * (X.T @ (predictions - y))

            # 更新参数
            self.theta[0] -= self.learning_rate * grad0
            self.theta[1:] -= self.learning_rate * grad_rest

            # 记录损失值
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            # 每500次迭代打印一次损失
            if i % 500 == 0 and i > 0:
                print(f"  迭代 {i}, 损失: {cost:.6f}")

    def predict(self, X):
        """
        使用训练好的模型进行预测

        参数:
            X: 特征值数组，形状 (m, n)

        返回:
            预测值数组
        """
        return X @ self.theta[1:] + self.theta[0]

    def get_params(self):
        """获取模型参数"""
        return self.theta

    def get_r2_score(self, X, y):
        """计算决定系数R²"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def get_mse(self, X, y):
        """计算均方误差"""
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse


def standardize(X):
    """
    标准化特征值（Z-score标准化）

    参数:
        X: 特征值数组，可以是1D或2D

    返回:
        标准化后的特征值, 均值, 标准差
    """
    if X.ndim == 1:
        mean = np.mean(X)
        std = np.std(X)
        X_std = (X - mean) / std
        return X_std, mean, std
    else:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_std = (X - mean) / std
        return X_std, mean, std


# ==================== 第二部分：一元线性回归分析 ====================

def experiment_univariate_iris():
    """
    实验1：IRIS数据集的一元线性回归分析
    分析花萼长度与花萼宽度、花瓣长度、花瓣宽度之间的一元线性关系
    """
    print("=" * 80)
    print("实验1：IRIS数据集一元线性回归分析")
    print("=" * 80)

    # 加载IRIS数据集
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    print("\n【数据集基本信息】")
    print(f"样本数量: {len(df)}")
    print(f"特征名称: {list(df.columns)}")
    print(f"目标变量: 花萼长度 (sepal length)")
    print("\n数据集前10行:")
    print(df.head(10))

    # 定义特征和目标
    # 目标变量：花萼长度
    y = df['sepal length (cm)'].values

    # 三个特征变量：花萼宽度、花瓣长度、花瓣宽度
    features = {
        'sepal width (cm)': '花萼宽度',
        'petal length (cm)': '花瓣长度',
        'petal width (cm)': '花瓣宽度'
    }

    results = {}

    # 为每个特征建立一元线性回归模型
    for feature_name, feature_cn in features.items():
        print(f"\n{'='*60}")
        print(f"分析：花萼长度 vs {feature_cn}")
        print(f"{'='*60}")

        X = df[feature_name].values  # 特征值

        # 数据标准化（加速梯度下降收敛）
        X_std, X_mean, X_std_val = standardize(X)

        # 创建并训练模型
        print("训练模型中...")
        model = LinearRegressionGD(learning_rate=0.1, n_iterations=2000)
        model.fit(X_std.reshape(-1, 1), y)

        # 获取模型参数（标准化后的参数）
        theta = model.get_params()
        theta0_std, theta1_std = theta[0], theta[1]

        # 转换回原始尺度的参数
        # 原始公式: y = theta0_orig + theta1_orig * X
        # 标准化后: y = theta0_std + theta1_std * ((X - mean)/std)
        # 整理得: y = (theta0_std - theta1_std*mean/std) + (theta1_std/std) * X
        theta1_orig = theta1_std / X_std_val
        theta0_orig = theta0_std - theta1_std * X_mean / X_std_val

        # 评估模型
        y_pred_std = model.predict(X_std.reshape(-1, 1))
        # 注意：由于使用的是标准化特征训练的模型，预测时也需要用标准化特征
        # 上面的预测已经是正确的

        mse = model.get_mse(X_std.reshape(-1, 1), y)
        r2 = model.get_r2_score(X_std.reshape(-1, 1), y)

        print(f"\n模型训练完成！")
        print(f"线性回归方程: 花萼长度 = {theta0_orig:.4f} + {theta1_orig:.4f} × {feature_cn}")
        print(f"模型评估指标:")
        print(f"  - 均方误差 (MSE): {mse:.6f}")
        print(f"  - 决定系数 (R²): {r2:.6f}")

        # 判断相关性方向
        if theta1_orig > 0:
            print(f"相关性: 正相关（{feature_cn}越大，花萼长度越大）")
        else:
            print(f"相关性: 负相关（{feature_cn}越大，花萼长度越小）")

        # 存储结果用于后续可视化
        results[feature_cn] = {
            'model': model,
            'X': X,
            'X_std': X_std,
            'X_mean': X_mean,
            'X_std_val': X_std_val,
            'theta0': theta0_orig,
            'theta1': theta1_orig,
            'mse': mse,
            'r2': r2,
            'y_pred': y_pred_std
        }

    # ==================== 可视化部分 ====================

    # 创建大图：4个子图，展示一元回归结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IRIS数据集一元线性回归分析', fontsize=16, fontweight='bold')

    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (feature_cn, result) in enumerate(results.items()):
        X = result['X']
        X_sorted_idx = np.argsort(X)
        X_sorted = X[X_sorted_idx]

        # 原始尺度下的预测值
        y_pred_orig = result['theta0'] + result['theta1'] * X_sorted

        # 子图1：散点图 + 拟合直线（左上区域）
        ax = axes[idx // 2, idx % 2]
        ax.scatter(X, y, alpha=0.7, c=colors[idx], edgecolors='white', s=80, label='实际数据')
        ax.plot(X_sorted, y_pred_orig, color='red', linewidth=2, label=f'拟合直线 (R²={result["r2"]:.3f})')
        ax.set_xlabel(f'{feature_cn} (cm)', fontsize=11)
        ax.set_ylabel('花萼长度 (cm)', fontsize=11)
        ax.set_title(f'花萼长度 vs {feature_cn}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 子图3（右上角）：损失收敛曲线对比
    ax3 = axes[1, 0]
    for idx, (feature_cn, result) in enumerate(results.items()):
        ax3.plot(range(len(result['model'].cost_history)), result['model'].cost_history,
                label=feature_cn, color=colors[idx], alpha=0.7)
    ax3.set_xlabel('迭代次数', fontsize=11)
    ax3.set_ylabel('损失值', fontsize=11)
    ax3.set_title('梯度下降损失收敛曲线对比', fontsize=12)
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 子图4（右下角）：R²对比柱状图
    ax4 = axes[1, 1]
    feature_names = list(results.keys())
    r2_values = [results[f]['r2'] for f in feature_names]
    bars = ax4.bar(feature_names, r2_values, color=colors, alpha=0.7)
    ax4.set_xlabel('特征变量', fontsize=11)
    ax4.set_ylabel('决定系数 R²', fontsize=11)
    ax4.set_title('一元回归模型R²对比', fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, val in zip(bars, r2_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 隐藏第6个子图（2x3布局中多余的那个）
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('iris_univariate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ==================== 补充：残差分析图 ====================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('一元线性回归残差分析', fontsize=14, fontweight='bold')

    for idx, (feature_cn, result) in enumerate(results.items()):
        X = result['X']
        y_pred = result['y_pred']
        residuals = y - y_pred

        ax = axes2[idx]
        ax.scatter(y_pred, residuals, alpha=0.7, c=colors[idx], edgecolors='white', s=60)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('预测值 (cm)', fontsize=11)
        ax.set_ylabel('残差 (cm)', fontsize=11)
        ax.set_title(f'{feature_cn} 残差图', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iris_univariate_residuals.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 输出分析结论
    print("\n" + "=" * 80)
    print("【一元线性回归分析结论】")
    print("=" * 80)
    print("1. 相关性分析:")
    for feature_cn, result in results.items():
        if result['theta1'] > 0:
            print(f"   - 花萼长度与{feature_cn}呈正相关 (系数={result['theta1']:.4f})")
        else:
            print(f"   - 花萼长度与{feature_cn}呈负相关 (系数={result['theta1']:.4f})")

    print("\n2. 模型拟合效果:")
    best_feature = max(results.items(), key=lambda x: x[1]['r2'])
    worst_feature = min(results.items(), key=lambda x: x[1]['r2'])
    print(f"   - 最佳预测变量: {best_feature[0]} (R²={best_feature[1]['r2']:.4f})")
    print(f"   - 最差预测变量: {worst_feature[0]} (R²={worst_feature[1]['r2']:.4f})")

    return results


# ==================== 第三部分：多元线性回归分析 ====================

def experiment_multivariate_iris():
    """
    实验2：IRIS数据集的多元线性回归分析
    使用花萼宽度、花瓣长度、花瓣宽度预测花萼长度
    """
    print("\n" + "=" * 80)
    print("实验2：IRIS数据集多元线性回归分析")
    print("=" * 80)

    # 加载数据
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # 特征矩阵 X：花萼宽度、花瓣长度、花瓣宽度
    X = df[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
    # 目标变量 y：花萼长度
    y = df['sepal length (cm)'].values

    print("\n【多元回归设定】")
    print(f"特征变量 (X):")
    print(f"  - x1: 花萼宽度 (sepal width)")
    print(f"  - x2: 花瓣长度 (petal length)")
    print(f"  - x3: 花瓣宽度 (petal width)")
    print(f"目标变量 (y): 花萼长度 (sepal length)")
    print(f"样本数量: {X.shape[0]}")

    # 数据标准化
    X_std, X_mean, X_std_val = standardize(X)

    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.2, random_state=42
    )
    print(f"\n数据划分: 训练集 {len(y_train)} 个样本, 测试集 {len(y_test)} 个样本")

    # 训练多元线性回归模型
    print("\n训练多元线性回归模型（梯度下降）...")
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=3000)
    model.fit(X_train, y_train)

    theta = model.get_params()
    print(f"\n模型训练完成！")
    print(f"模型参数:")
    print(f"  theta0 (截距): {theta[0]:.6f}")
    print(f"  theta1 (花萼宽度系数): {theta[1]:.6f}")
    print(f"  theta2 (花瓣长度系数): {theta[2]:.6f}")
    print(f"  theta3 (花瓣宽度系数): {theta[3]:.6f}")

    # 获取原始尺度的参数（用于解释）
    # 对于多元回归，原始尺度参数转换：theta_orig[0] = theta[0] - sum(theta[i] * mean[i]/std[i])
    #                            theta_orig[i] = theta[i] / std[i-1]
    theta_orig = np.zeros(len(theta))
    theta_orig[0] = theta[0]
    for i in range(1, len(theta)):
        theta_orig[i] = theta[i] / X_std_val[i-1]
        theta_orig[0] -= theta[i] * X_mean[i-1] / X_std_val[i-1]

    print(f"\n原始尺度回归方程:")
    print(f"花萼长度 = {theta_orig[0]:.4f} + {theta_orig[1]:.4f}×花萼宽度 + {theta_orig[2]:.4f}×花瓣长度 + {theta_orig[3]:.4f}×花瓣宽度")

    # 模型评估
    train_mse = model.get_mse(X_train, y_train)
    train_r2 = model.get_r2_score(X_train, y_train)
    test_mse = model.get_mse(X_test, y_test)
    test_r2 = model.get_r2_score(X_test, y_test)

    print(f"\n【模型评估】")
    print(f"训练集:")
    print(f"  - 均方误差 (MSE): {train_mse:.6f}")
    print(f"  - 决定系数 (R²): {train_r2:.6f}")
    print(f"测试集:")
    print(f"  - 均方误差 (MSE): {test_mse:.6f}")
    print(f"  - 决定系数 (R²): {test_r2:.6f}")

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all_pred = model.predict(X_std)

    # ==================== 可视化部分 ====================
    fig = plt.figure(figsize=(16, 12))

    # 子图1：损失收敛曲线
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(range(len(model.cost_history)), model.cost_history, 'b-', linewidth=1.5)
    ax1.set_xlabel('迭代次数', fontsize=11)
    ax1.set_ylabel('损失值', fontsize=11)
    ax1.set_title('多元回归梯度下降收敛曲线', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 子图2：真实值 vs 预测值（全部数据）
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(y, y_all_pred, alpha=0.6, c='steelblue', edgecolors='white', s=70)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='理想线 (y=x)')
    ax2.set_xlabel('真实花萼长度 (cm)', fontsize=11)
    ax2.set_ylabel('预测花萼长度 (cm)', fontsize=11)
    ax2.set_title(f'真实值 vs 预测值 (R²={test_r2:.3f})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：残差分布
    residuals = y - y_all_pred
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(y_all_pred, residuals, alpha=0.6, c='steelblue', edgecolors='white', s=70)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('预测值 (cm)', fontsize=11)
    ax3.set_ylabel('残差 (cm)', fontsize=11)
    ax3.set_title('残差图', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 子图4：残差直方图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.set_xlabel('残差 (cm)', fontsize=11)
    ax4.set_ylabel('频数', fontsize=11)
    ax4.set_title('残差分布直方图', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 子图5：参数重要性
    ax5 = fig.add_subplot(2, 3, 5)
    feature_names = ['花萼宽度', '花瓣长度', '花瓣宽度']
    param_importance = np.abs(theta[1:])  # 标准化后的系数绝对值表示重要性
    bars = ax5.bar(feature_names, param_importance, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax5.set_xlabel('特征变量', fontsize=11)
    ax5.set_ylabel('参数重要性 (标准化系数绝对值)', fontsize=11)
    ax5.set_title('特征重要性分析', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, param_importance):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 子图6：训练集 vs 测试集 R² 对比
    ax6 = fig.add_subplot(2, 3, 6)
    metrics = ['训练集 R²', '测试集 R²']
    values = [train_r2, test_r2]
    colors_plot = ['#2ca02c', '#ff7f0e']
    bars = ax6.bar(metrics, values, color=colors_plot, alpha=0.7)
    ax6.set_ylabel('R² 值', fontsize=11)
    ax6.set_title('模型泛化能力评估', fontsize=12)
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('IRIS数据集多元线性回归分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('iris_multivariate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 输出分析结论
    print("\n" + "=" * 80)
    print("【多元线性回归分析结论】")
    print("=" * 80)
    print("1. 回归方程参数解读:")
    print(f"   - 花萼宽度系数: {theta_orig[1]:.4f} (花萼宽度每增加1cm，花萼长度变化{theta_orig[1]:.4f}cm)")
    print(f"   - 花瓣长度系数: {theta_orig[2]:.4f} (花瓣长度每增加1cm，花萼长度变化{theta_orig[2]:.4f}cm)")
    print(f"   - 花瓣宽度系数: {theta_orig[3]:.4f} (花瓣宽度每增加1cm，花萼长度变化{theta_orig[3]:.4f}cm)")

    print("\n2. 模型性能:")
    print(f"   - 训练集 R²: {train_r2:.6f}")
    print(f"   - 测试集 R²: {test_r2:.6f}")
    if test_r2 > 0.8:
        print(f"   - 模型表现: 优秀 (R² > 0.8)")
    elif test_r2 > 0.6:
        print(f"   - 模型表现: 良好 (R² > 0.6)")
    else:
        print(f"   - 模型表现: 一般 (R² < 0.6)")

    print("\n3. 特征重要性排序:")
    feature_importance = sorted(zip(feature_names, param_importance), key=lambda x: -x[1])
    for i, (name, imp) in enumerate(feature_importance, 1):
        print(f"   {i}. {name}: {imp:.4f}")

    return model


# ==================== 第四部分：一元 vs 多元对比 ====================

def compare_models(univariate_results, multivariate_model, X_std, y):
    """
    对比一元回归和多元回归的效果
    """
    print("\n" + "=" * 80)
    print("实验3：一元回归 vs 多元回归 效果对比")
    print("=" * 80)

    # 收集一元回归的R²
    univariate_r2 = {name: result['r2'] for name, result in univariate_results.items()}

    # 多元回归的R²
    multivariate_r2 = multivariate_model.get_r2_score(X_std, y)

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('一元回归 vs 多元回归 效果对比', fontsize=14, fontweight='bold')

    # 子图1：R²对比柱状图
    ax1 = axes[0]
    models = list(univariate_r2.keys()) + ['多元回归']
    r2_values = list(univariate_r2.values()) + [multivariate_r2]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(models, r2_values, color=colors[:len(models)], alpha=0.7)
    ax1.set_ylabel('决定系数 R²', fontsize=12)
    ax1.set_title('模型拟合效果对比', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 子图2：R²提升效果
    ax2 = axes[1]
    best_univariate_r2 = max(univariate_r2.values())
    improvement = multivariate_r2 - best_univariate_r2
    improvement_pct = (improvement / best_univariate_r2) * 100

    categories = ['最佳一元模型', '多元模型']
    values = [best_univariate_r2, multivariate_r2]
    bars = ax2.bar(categories, values, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_ylabel('决定系数 R²', fontsize=12)
    ax2.set_title(f'多元回归提升效果 (+{improvement_pct:.1f}%)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 添加箭头表示提升
    if improvement > 0:
        ax2.annotate(f'+{improvement:.4f}', xy=(1, multivariate_r2), xytext=(0.7, multivariate_r2 - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=11, color='red')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n【对比分析结果】")
    print(f"最佳一元回归模型: {max(univariate_r2.items(), key=lambda x: x[1])[0]} (R²={best_univariate_r2:.4f})")
    print(f"多元回归模型 R²: {multivariate_r2:.4f}")
    print(f"多元回归相比最佳一元模型提升: {improvement:.4f} ({improvement_pct:.2f}%)")

    if multivariate_r2 > best_univariate_r2:
        print("\n结论: 多元线性回归模型优于所有一元线性回归模型")
        print("      使用多个特征共同预测能够更准确地估计花萼长度")
    else:
        print("\n结论: 一元线性回归模型效果更好，可能存在过拟合")

    return improvement


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 80)
    print("                    实验一：一元/多元线性回归算法Python实现")
    print("                    数据集：IRIS（鸢尾花数据集）")
    print("=" * 80)

    # 实验1：一元线性回归分析
    univariate_results = experiment_univariate_iris()

    # 实验2：多元线性回归分析
    multivariate_model = experiment_multivariate_iris()

    # 加载数据用于对比
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    X_std, _, _ = standardize(df[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values)
    y = df['sepal length (cm)'].values

    # 实验3：对比分析
    compare_models(univariate_results, multivariate_model, X_std, y)

    print("\n" + "=" * 80)
    print("                        所有实验完成！")
    print("=" * 80)