import pandas as pd
import numpy as np
import os
import time
from compute_plif import compute_simple_plif

df = pd.read_csv("pdbbind_data.csv")
refined_dir = "refined-set"

print(f"开始处理 {len(df)} 个复合物...")
t0 = time.time()

feature_list = []
valid_indices = []

for idx, row in df.iterrows():
    pdb = row['pdb']
    protein_path = os.path.join(refined_dir, pdb, f"{pdb}_protein.pdb")
    lig_path = os.path.join(refined_dir, pdb, f"{pdb}_ligand.{row['ligand_suffix']}")

    if not os.path.exists(lig_path):
        continue

    fp = compute_simple_plif(protein_path, lig_path)
    if fp is not None:
        feature_list.append(fp)
        valid_indices.append(idx)

    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  已处理 {idx + 1}/{len(df)}，有效: {len(feature_list)}，耗时: {elapsed:.1f}s")

X = np.array(feature_list)
df_valid = df.iloc[valid_indices].copy()
y = df_valid['pK'].values

print(f"\n完成！耗时: {time.time() - t0:.1f}s")
print(f"特征矩阵形状: {X.shape}")
print(f"标签范围: {y.min():.2f} ~ {y.max():.2f}")
print(f"特征均值: {X.mean(axis=0).round(3)}")
print(f"特征名称: [疏水, H供体, H受体, π-π, 盐桥, 金属]")

np.save("X_features.npy", X)
np.save("y_labels.npy", y)
df_valid.to_csv("pdbbind_valid.csv", index=False)
print("\n已保存 X_features.npy, y_labels.npy, pdbbind_valid.csv")
