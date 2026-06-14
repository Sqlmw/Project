"""
Step 7：RF 超参数调优，诊断模型端瓶颈
目的：确认过拟合程度，判断 Spearman 提升空间是否在模型端
"""
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, spearmanr

X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# ============= 1. 超参数搜索 =============
print("RF 超参数搜索 (RandomizedSearchCV, 50组)...")

# 参数空间：树数量均匀分布、最大深度候选列表、叶子/分裂最小样本
param_dist = {
    'n_estimators': randint(100, 600),
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_leaf': randint(1, 30),
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2', 1.0],
}

# 随机抽 50 组参数，每组 5 折 CV，以 MAE 最低为优化目标
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5, random_state=42,
    scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
)
search.fit(X, y)

print(f"\n最佳参数: {search.best_params_}")
print(f"最佳 CV MAE: {-search.best_score_:.3f}")

# ============= 2. 对比验证：默认 vs 调优 =============
print("\n对比 (5次重复, Random split):")
indices = np.arange(len(X))

for seed in range(5):
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    # 默认 RF：200 棵树，深度不限 → 容易过拟合
    rf_default = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    rf_default.fit(X[train_idx], y[train_idx])
    pred_d = rf_default.predict(X[test_idx])
    rho_d, _ = spearmanr(y[test_idx], pred_d)
    mae_d = np.mean(np.abs(pred_d - y[test_idx]))

    # 调优 RF：加深度和分裂限制 → 减少过拟合
    rf_tuned = RandomForestRegressor(**search.best_params_, n_jobs=-1, random_state=seed)
    rf_tuned.fit(X[train_idx], y[train_idx])
    pred_t = rf_tuned.predict(X[test_idx])
    rho_t, _ = spearmanr(y[test_idx], pred_t)
    mae_t = np.mean(np.abs(pred_t - y[test_idx]))

    # 训练集 R²：过拟合的直观指标（太接近 1.0 说明模型记忆了训练集）
    train_r2_d = rf_default.score(X[train_idx], y[train_idx])
    train_r2_t = rf_tuned.score(X[train_idx], y[train_idx])

    print(f"  seed={seed}: default(train R2={train_r2_d:.3f},test rho={rho_d:.3f},MAE={mae_d:.3f}) "
          f"| tuned(train R2={train_r2_t:.3f},test rho={rho_t:.3f},MAE={mae_t:.3f})")
