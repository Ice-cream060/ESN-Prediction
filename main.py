# ========== 第一步：导入必要的包和模块 ==========
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import to_forecasting

from data import quiet_heart_rate, sleep_score

rpy.set_seed(42)


# ========== 第二步：定义数据处理类 ==========
class DataPipeline:
    """负责数据的导入、清洗、归一化以及时序数据集的划分"""

    def __init__(self, heart_rate, sleep, test_size=0.2, forecast=1):
        self.heart_rate = heart_rate
        self.sleep = sleep
        self.test_size = test_size
        self.forecast = forecast
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def process_data(self):
        # 1. 统一数据长度
        min_length = min(len(self.heart_rate), len(self.sleep))
        df = pd.DataFrame({
            'HeartRate': self.heart_rate[:min_length],
            'SleepScore': self.sleep[:min_length]
        })

        # 2. 替换0值为线性插值
        df.replace(0, np.nan, inplace=True).interpolate(method='linear', inplace=True)
        df.bfill(inplace=True).ffill(inplace=True)
        cleaned_data = df.values

        # 3. 数据归一化
        scaled_data = self.scaler.fit_transform(cleaned_data)

        # 4. 划分为时序监督学习格式 (X_today -> Y_tomorrow)
        X_all, Y_all = to_forecasting(scaled_data, forecast=self.forecast)

        # 5. 划分训练集和测试集
        train_len = int(len(X_all) * (1 - self.test_size))
        dataset = {
            'X_train': X_all[:train_len],
            'Y_train': Y_all[:train_len],
            'X_test': X_all[train_len:],
            'Y_test': Y_all[train_len:],
            'X_all': X_all,
            'Y_all': Y_all,
            'last_today_input': scaled_data[-1]  # 用于预测明天的最后一条特征
        }
        return dataset

    def inverse_transform(self, data):
        """将预测结果反归一化回真实生理数值"""
        return self.scaler.inverse_transform(data)


# ========== 第三步：定义模型类 ==========
class ESNPredictor:
    """封装回声状态网络(ESN)的初始化、训练与预测逻辑"""

    def __init__(self, units=100, sr=0.999999, lr=0.1, rc_connectivity=0.1, ridge=1.0):
        self.reservoir = Reservoir(
            units=units,
            sr=sr,
            lr=lr,
            rc_connectivity=rc_connectivity
        )
        self.readout = Ridge(ridge=ridge)
        self.model = self.reservoir >> self.readout

    def train(self, X, Y, warmup=10):
        self.model = self.model.fit(X, Y, warmup=warmup)

    def predict(self, X):
        return self.model.run(X)

    def predict_single_step(self, x):
        """进行单步推演预测"""
        return self.model(x)


# ========== 第四步：定义评估与可视化类 ==========
class ExperimentEvaluator:
    """负责模型误差指标的计算与学术级结果可视化"""

    @staticmethod
    def print_metrics(y_true, y_pred):
        # 计算心率指标 (Index 0)
        mse_hr = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        rmse_hr = np.sqrt(mse_hr)
        mae_hr = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        r2_hr = r2_score(y_true[:, 0], y_pred[:, 0])
        var_hr = np.var(y_pred[:, 0] - y_true[:, 0])

        # 计算睡眠分数指标 (Index 1)
        mse_ss = mean_squared_error(y_true[:, 1], y_pred[:, 1])
        rmse_ss = np.sqrt(mse_ss)
        mae_ss = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        r2_ss = r2_score(y_true[:, 1], y_pred[:, 1])
        var_ss = np.var(y_pred[:, 1] - y_true[:, 1])

        print("\n" + "=" * 20 + " 模型性能评价报告 " + "=" * 20)
        print(f"{'指标':<15} | {'心率 (Heart Rate)':<18} | {'睡眠分数 (Sleep Score)':<18}")
        print("-" * 60)
        print(f"{'MSE (均方误差)':<12}   | {mse_hr:<18.4f}   | {mse_ss:<18.4f}")
        print(f"{'RMSE (均方根误差)':<11}   | {rmse_hr:<18.4f}  | {rmse_ss:<18.4f}")
        print(f"{'MAE (平均绝对误差)':<11}  | {mae_hr:<18.4f}   | {mae_ss:<18.4f}")
        print(f"{'R² (决定系数)':<13}  | {r2_hr:<18.4f}    | {r2_ss:<18.4f}")
        print(f"{'残差方差':<15}  | {var_hr:<18.4f}   | {var_ss:<18.4f}")
        print("=" * 58)
        print("注：R² 越接近 1 表示模型拟合效果越好；RMSE 的单位与原数据一致，更具直观参考价值。\n")

    @staticmethod
    def plot_academic_results(y_true, y_pred, save_path="ESN_Prediction_Academic.png"):
        # 全局学术字体与样式配置
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
        plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 1.2

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True, constrained_layout=True)

        # --- 子图 (a): 心率预测 ---
        ax1.plot(y_true[:, 0], color='black', linestyle='-', linewidth=1.5,
                 marker='o', markersize=5, markerfacecolor='white', label='Actual Observation', zorder=2)
        ax1.plot(y_pred[:, 0], color='#B22222', linestyle='--', linewidth=1.5,
                 marker='s', markersize=5, markeredgecolor='white', markeredgewidth=0.8, label='ESN Prediction',
                 zorder=3)

        ax1.set_ylabel('Resting Heart Rate (BPM)', fontweight='bold')
        ax1.set_title('(a) Comparison of Actual and Predicted Heart Rate', loc='left', pad=12, fontweight='bold')

        y1_max = max(y_true[:, 0].max(), y_pred[:, 0].max())
        y1_min = min(y_true[:, 0].min(), y_pred[:, 0].min())
        ax1.set_ylim(y1_min - 2, y1_max + (y1_max - y1_min) * 0.35)
        ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC', framealpha=1.0)

        # --- 子图 (b): 睡眠分数预测 ---
        ax2.plot(y_true[:, 1], color='black', linestyle='-', linewidth=1.5,
                 marker='o', markersize=5, markerfacecolor='white', label='Actual Observation', zorder=2)
        ax2.plot(y_pred[:, 1], color='#000080', linestyle='--', linewidth=1.5,
                 marker='^', markersize=6, markeredgecolor='white', markeredgewidth=0.8, label='ESN Prediction',
                 zorder=3)

        ax2.set_xlabel('Time Steps (Days in Test Set)', fontweight='bold')
        ax2.set_ylabel('Sleep Score', fontweight='bold')
        ax2.set_title('(b) Comparison of Actual and Predicted Sleep Score', loc='left', pad=12, fontweight='bold')

        y2_max = max(y_true[:, 1].max(), y_pred[:, 1].max())
        y2_min = min(y_true[:, 1].min(), y_pred[:, 1].min())
        ax2.set_ylim(y2_min - 2, y2_max + (y2_max - y2_min) * 0.35)
        ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC', framealpha=1.0)

        # 全局坐标轴与网格精调
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='in', length=5, width=1.2, bottom=True, left=True)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
            ax.xaxis.grid(False)

        plt.savefig(save_path, dpi=600)
        print(f"学术图表已保存至: {save_path}")
        plt.show()


# ========== 第五步：主程序入口 ==========
def main():
    # 1. 实例化数据管道并处理数据
    pipeline = DataPipeline(quiet_heart_rate, sleep_score, test_size=0.2)
    dataset = pipeline.process_data()

    # 2. 实例化模型并在测试集上进行验证
    print("\n>>> 阶段一：模型验证 (80% 训练，20% 测试) <<<")
    eval_model = ESNPredictor(units=312, sr=0.99, lr=0.74, rc_connectivity=0.3, ridge=0.1)
    print("开始训练验证模型...")
    eval_model.train(dataset['X_train'], dataset['Y_train'], warmup=10)

    # 预测并反归一化验证集结果
    predictions_scaled = eval_model.predict(dataset['X_test'])
    predictions = pipeline.inverse_transform(predictions_scaled)
    Y_test_real = pipeline.inverse_transform(dataset['Y_test'])

    # 计算指标并绘图
    Evaluator = ExperimentEvaluator()
    Evaluator.print_metrics(Y_test_real, predictions)
    Evaluator.plot_academic_results(Y_test_real, predictions)

    # 3. 基于全量数据预测“明天”
    print("-" * 30)
    print(">>> 阶段二：部署应用 (基于全量数据预测明天) <<<")
    deploy_model = ESNPredictor(units=312, sr=0.99, lr=0.74, rc_connectivity=0.3, ridge=0.1)
    print("开始训练部署模型...")
    deploy_model.train(dataset['X_all'], dataset['Y_all'], warmup=10)

    tomorrow_pred_scaled = deploy_model.predict_single_step(dataset['last_today_input'])
    tomorrow_final = pipeline.inverse_transform(tomorrow_pred_scaled.reshape(1, -1))

    print(f"\n基于全量 {len(dataset['X_all']) + 1} 天数据的预测：")
    print(f"明天预测静息心率: {tomorrow_final[0, 0]:.1f} BPM")
    print(f"明天预测睡眠分数: {tomorrow_final[0, 1]:.1f} 分")
    print("-" * 30)


if __name__ == "__main__":
    main()
