"""
Step 1：解析 PDBbind 索引文件，筛选文件齐全的复合物
输入: refined-set/index/INDEX_refined_data.2020
输出: pdbbind_data.csv（仅保留蛋白+配体文件均存在的复合物）
"""
import os
import pandas as pd

# PDBbind refined-set 的索引文件和数据根目录
index_path = "refined-set/index/INDEX_refined_data.2020"
refined_dir = "refined-set"

records = []
with open(index_path, 'r') as f:
    for line in f:
        # 跳过注释行和空行
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        # 每行按空白字符切分，前 5 列为固定字段
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        pdb_code = parts[0]          # PDB 编号，如 "1a30"
        resolution = float(parts[1]) # 晶体结构分辨率 (Å)
        year = int(parts[2])         # 测定年份
        pk = float(parts[3])         # -log(Kd/Ki)，结合亲和力
        affinity_type = parts[4]     # 亲和力数据类型（Kd 或 Ki）

        # 拼接蛋白和配体的预期文件路径
        # PDBbind 目录结构: refined-set/{PDB编号}/{PDB编号}_{类型}.{后缀}
        protein_file = os.path.join(refined_dir, pdb_code, f"{pdb_code}_protein.pdb")
        ligand_mol2 = os.path.join(refined_dir, pdb_code, f"{pdb_code}_ligand.mol2")
        ligand_sdf = os.path.join(refined_dir, pdb_code, f"{pdb_code}_ligand.sdf")

        # 检查文件是否真实存在（蛋白必须有，配体 mol2 优先，没有则用 sdf）
        if os.path.exists(protein_file) and (os.path.exists(ligand_mol2) or os.path.exists(ligand_sdf)):
            file_ok = True
            lig_suffix = 'mol2' if os.path.exists(ligand_mol2) else 'sdf'
        else:
            file_ok = False
            lig_suffix = None

        # 收集该复合物的所有信息
        records.append({
            'pdb': pdb_code,
            'resolution': resolution,
            'year': year,
            'pK': pk,
            'affinity_type': affinity_type,
            'file_ok': file_ok,
            'ligand_suffix': lig_suffix  # 记录配体格式，step2 需要用它拼接路径
        })

# 转 DataFrame，筛出文件齐全的复合物，保存
df = pd.DataFrame(records)
print(f"总记录: {len(df)}，文件齐全: {df['file_ok'].sum()}")
df_clean = df[df['file_ok']].copy()
print(f"最终有效复合物数: {len(df_clean)}")
df_clean.to_csv("pdbbind_data.csv", index=False)
print("已保存 pdbbind_data.csv")
