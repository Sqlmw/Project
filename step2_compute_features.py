"""
Step 2：批量计算 PLIF 指纹，构建特征矩阵和标签向量
输入: pdbbind_data.csv + refined-set/ 目录
输出: X_features.npy (特征矩阵), y_labels.npy (标签), pdbbind_valid.csv
"""
import pandas as pd
import numpy as np
import os
import time
from compute_plif import compute_simple_plif  # 核心函数，计算单个复合物的 5 维指纹

# 加载 step1 产出的有效复合物列表
df = pd.read_csv("pdbbind_data.csv")
refined_dir = "refined-set"

print(f"开始处理 {len(df)} 个复合物...")
t0 = time.time()

feature_list = []   # 收集每个复合物的 5 维指纹
valid_indices = []  # 记录成功计算的复合物在 df 中的行号（用于对齐标签）

# 遍历每个复合物，逐个提取 PLIF 指纹
for idx, row in df.iterrows():
    pdb = row['pdb']

    # 拼接蛋白和配体的文件路径
    protein_path = os.path.join(refined_dir, pdb, f"{pdb}_protein.pdb")
    lig_path = os.path.join(refined_dir, pdb, f"{pdb}_ligand.{row['ligand_suffix']}")

    # 安全检查：配体文件不存在则跳过
    if not os.path.exists(lig_path):
        continue

    # 调用核心函数计算 5 维 PLIF 指纹
    fp = compute_simple_plif(protein_path, lig_path)
    if fp is not None:
        feature_list.append(fp)
        valid_indices.append(idx)

    # 每 500 个打印一次进度
    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  已处理 {idx + 1}/{len(df)}，有效: {len(feature_list)}，耗时: {elapsed:.1f}s")

# 组装特征矩阵 X 和标签向量 y
X = np.array(feature_list)                          # shape: (n_valid, 5)
df_valid = df.iloc[valid_indices].copy()             # 成功计算的复合物元数据
y = df_valid['pK'].values                            # 标签 = 实验测定的 pK 值

print(f"\n完成！耗时: {time.time() - t0:.1f}s")
print(f"特征矩阵形状: {X.shape}")
print(f"标签范围: {y.min():.2f} ~ {y.max():.2f}")
print(f"特征均值: {X.mean(axis=0).round(3)}")
print(f"特征名称: [疏水, H供体, H受体, π-π, 盐桥]")

# 保存后续步骤所需的核心文件
np.save("X_features.npy", X)
np.save("y_labels.npy", y)
df_valid.to_csv("pdbbind_valid.csv", index=False)
print("\n已保存 X_features.npy, y_labels.npy, pdbbind_valid.csv")
