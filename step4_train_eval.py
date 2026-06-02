"""
第4天：对每种划分分别训练RF和MLP，在各自测试集上评估泛化性能

核心改进：每种划分用自己的训练集（互补集），避免信息泄漏。
"""
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

# 加载数据
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
df = pd.read_csv("pdbbind_valid.csv")

with open('splits.pkl', 'rb') as f:
    splits = pickle.load(f)

indices = np.arange(len(X))
print(f"总样本: {len(X)}, 特征维度: {X.shape[1]}, pK范围: [{y.min():.2f}, {y.max():.2f}]")


def evaluate(model, X_test, y_test):
    """计算三个评估指标"""
    pred = model.predict(X_test)
    rho, _ = spearmanr(y_test, pred)
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    return {'Spearman': rho, 'MAE': mae, 'RMSE': rmse}


# ---- 对每种划分分别训练+评估 ----
print("\n========== 评估结果 ==========")
results = {}
models = {}

for name, test_idx in splits.items():
    if len(test_idx) == 0:
        continue

    # 互补集 = 全部 - 测试集
    train_idx = np.setdiff1d(indices, test_idx)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n--- {name} ---")
    print(f"  训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

    # 训练 RF
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_train_r2 = rf.score(X_train, y_train)
    rf_res = evaluate(rf, X_test, y_test)
    print(f"  RF:  train R2={rf_train_r2:.3f}, test Spearman={rf_res['Spearman']:.3f}, "
          f"MAE={rf_res['MAE']:.3f}, RMSE={rf_res['RMSE']:.3f}")

    # 训练 MLP
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_train_r2 = mlp.score(X_train, y_train)
    mlp_res = evaluate(mlp, X_test, y_test)
    print(f"  MLP: train R2={mlp_train_r2:.3f}, test Spearman={mlp_res['Spearman']:.3f}, "
          f"MAE={mlp_res['MAE']:.3f}, RMSE={mlp_res['RMSE']:.3f}")

    results[name] = {'RF': rf_res, 'MLP': mlp_res}
    models[name] = {'RF': rf, 'MLP': mlp}

# ---- 保存结果 ----
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 只保存 Random 划分的模型（后续分析用）
import joblib
if 'Random' in models:
    joblib.dump(models['Random']['RF'], 'rf_model.pkl')
    joblib.dump(models['Random']['MLP'], 'mlp_model.pkl')

print("\n========== 汇总 ==========")
print(f"{'Split':<15} {'RF Spearman':>12} {'RF MAE':>8} {'RF RMSE':>8}  |  {'MLP Spearman':>13} {'MLP MAE':>9} {'MLP RMSE':>9}")
print("-" * 90)
for name in results:
    rf_r = results[name]['RF']
    mlp_r = results[name]['MLP']
    print(f"{name:<15} {rf_r['Spearman']:12.3f} {rf_r['MAE']:8.3f} {rf_r['RMSE']:8.3f}  |  "
          f"{mlp_r['Spearman']:13.3f} {mlp_r['MAE']:9.3f} {mlp_r['RMSE']:9.3f}")

print("\n已保存: results.json")
