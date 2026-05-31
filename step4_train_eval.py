"""
第4天：训练随机森林和MLP，在四种划分上评估泛化性能
"""
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# 加载数据
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
df = pd.read_csv("pdbbind_valid.csv")

with open('splits.pkl', 'rb') as f:
    splits = pickle.load(f)

# 用随机划分的训练集来训练模型
indices = np.arange(len(X))
train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
X_train, y_train = X[train_idx], y[train_idx]

print(f"训练集: {len(X_train)} 个样本")
print(f"X shape: {X.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")

# ---- 训练随机森林 ----
print("\n训练 Random Forest...")
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print(f"  RF train R^2: {rf.score(X_train, y_train):.3f}")

# ---- 训练 MLP ----
print("训练 MLP...")
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
print(f"  MLP train R^2: {mlp.score(X_train, y_train):.3f}")

# ---- 评估函数 ----
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    rho, p = spearmanr(y_test, pred)
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    return {'Spearman': rho, 'MAE': mae, 'RMSE': rmse}

# ---- 在所有划分上评估 ----
print("\n========== 评估结果 ==========")
results = {}
for name, test_idx in splits.items():
    if len(test_idx) == 0:
        continue
    X_test, y_test = X[test_idx], y[test_idx]
    rf_res = evaluate(rf, X_test, y_test)
    mlp_res = evaluate(mlp, X_test, y_test)
    results[name] = {'RF': rf_res, 'MLP': mlp_res}
    print(f"\n--- {name} (n={len(test_idx)}) ---")
    print(f"  RF:  Spearman={rf_res['Spearman']:.3f}, MAE={rf_res['MAE']:.3f}, RMSE={rf_res['RMSE']:.3f}")
    print(f"  MLP: Spearman={mlp_res['Spearman']:.3f}, MAE={mlp_res['MAE']:.3f}, RMSE={mlp_res['RMSE']:.3f}")

# 保存详细结果
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ---- 保存模型 ----
import joblib
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(mlp, 'mlp_model.pkl')

print("\n结果已保存: results.json, rf_model.pkl, mlp_model.pkl")
