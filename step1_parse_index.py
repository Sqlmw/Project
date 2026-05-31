import os
import pandas as pd

# 你的实际路径
index_path = "refined-set/index/INDEX_refined_data.2020"
refined_dir = "refined-set"

records = []
with open(index_path, 'r') as f:
    for line in f:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        pdb_code = parts[0]
        resolution = float(parts[1])
        year = int(parts[2])
        pk = float(parts[3])
        affinity_type = parts[4]

        protein_file = os.path.join(refined_dir, pdb_code, f"{pdb_code}_protein.pdb")
        ligand_mol2 = os.path.join(refined_dir, pdb_code, f"{pdb_code}_ligand.mol2")
        ligand_sdf = os.path.join(refined_dir, pdb_code, f"{pdb_code}_ligand.sdf")
        if os.path.exists(protein_file) and (os.path.exists(ligand_mol2) or os.path.exists(ligand_sdf)):
            file_ok = True
            lig_suffix = 'mol2' if os.path.exists(ligand_mol2) else 'sdf'
        else:
            file_ok = False
            lig_suffix = None

        records.append({
            'pdb': pdb_code,
            'resolution': resolution,
            'year': year,
            'pK': pk,
            'affinity_type': affinity_type,
            'file_ok': file_ok,
            'ligand_suffix': lig_suffix
        })

df = pd.DataFrame(records)
print(f"总记录: {len(df)}，文件齐全: {df['file_ok'].sum()}")
df_clean = df[df['file_ok']].copy()
print(f"最终有效复合物数: {len(df_clean)}")
df_clean.to_csv("pdbbind_data.csv", index=False)
print("已保存 pdbbind_data.csv")
