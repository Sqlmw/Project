"""
Permutation Importance：打乱特征列，测量 Spearman 下降量。
下降越多 → 该特征越重要。
"""
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

# 确保工作目录在数据文件所在的上级目录
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

X = np.load("X_features.npy")
y = np.load("y_labels.npy")
with open('splits.pkl', 'rb') as f:
    splits = pickle.load(f)

names = ['hydrophobic', 'H-donor', 'H-acceptor', 'pi-pi', 'salt_bridge']
indices = np.arange(len(X))
N_REPEATS = 5

print("Permutation Importance (Spearman drop, 5 repeats):")
print("=" * 75)

for split_name, test_idx in splits.items():
    train_idx = np.setdiff1d(indices, test_idx)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    rf = RandomForestRegressor(n_estimators=400, max_depth=15,
                               min_samples_leaf=3, min_samples_split=6,
                               max_features='sqrt', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    base_pred = rf.predict(X_test)
    base_rho, _ = spearmanr(y_test, base_pred)

    print(f"\n--- {split_name} (base Spearman={base_rho:.3f}) ---")

    for j, name in enumerate(names):
        drops = []
        for seed in range(N_REPEATS):
            X_perm = X_test.copy()
            rng = np.random.RandomState(seed)
            rng.shuffle(X_perm[:, j])
            perm_pred = rf.predict(X_perm)
            perm_rho, _ = spearmanr(y_test, perm_pred)
            drops.append(base_rho - perm_rho)
        mean_drop = np.mean(drops)
        std_drop = np.std(drops)
        bar = '+' * max(0, int(mean_drop * 100))
        print(f"  {name:<14} drop={mean_drop:+.4f} (std={std_drop:.4f}) {bar}")
