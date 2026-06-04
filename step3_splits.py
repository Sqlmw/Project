"""
第3天：设计四种数据划分策略
1. 随机划分（基线）
2. 配体化学相似度划分（Morgan指纹 + Tanimoto聚类）
3. 蛋白序列相似度划分（k-mer聚类）
4. 结合模式划分（PLIF聚类）
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.cluster import KMeans
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import pickle
import os
import warnings

# 抑制 RDKit 警告
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

X = np.load("X_features.npy")
y = np.load("y_labels.npy")
df = pd.read_csv("pdbbind_valid.csv")
refined_dir = "refined-set"

indices = np.arange(len(X))
splits = {}

# ==================== 1. 随机划分 ====================
print("1/4 随机划分...")
train_idx, test_idx_random = train_test_split(indices, test_size=0.2, random_state=42)
splits['Random'] = test_idx_random
print(f"   测试集: {len(test_idx_random)}")

# ==================== 2. 配体化学相似度划分 ====================
print("2/4 配体化学相似度划分 (Morgan指纹 + Tanimoto聚类)...")

from rdkit.Chem import AllChem

def get_morgan_fp(pdb, lig_suffix):
    """读取配体，返回 Morgan 指纹（ECFP4, 2048-bit）"""
    lig_path = os.path.join(refined_dir, pdb, f"{pdb}_ligand.{lig_suffix}")
    try:
        if lig_suffix == 'mol2':
            mol = Chem.MolFromMol2File(lig_path, sanitize=True, removeHs=True)
        else:
            mol = Chem.MolFromMolFile(lig_path, sanitize=True, removeHs=True)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except:
        return None

# 计算所有配体的 Morgan 指纹
print("   计算配体 Morgan 指纹...")
fps = {}
for idx, row in df.iterrows():
    fp = get_morgan_fp(row['pdb'], row['ligand_suffix'])
    if fp is not None:
        fps[idx] = fp

# 计算无效（无法读取配体）的设独立组
df['lig_cluster'] = df.index.map(lambda i: f"lig_{i}")

# Tanimoto 贪心聚类
from rdkit.DataStructs import TanimotoSimilarity

print("   运行 Tanimoto 聚类 (threshold=0.5)...")
lig_clusters = {}       # cluster_id -> [indices]
lig_rep_fps = {}        # cluster_id -> representative fingerprint
lig_cluster_of = {}     # df_index -> cluster_id

valid_indices = list(fps.keys())
cluster_id = 0

# 按指纹有效数的降序排列（等价于按配体大小/复杂度，近似的）
for idx in valid_indices:
    fp = fps[idx]
    assigned = False
    for cid in list(lig_clusters.keys()):
        if TanimotoSimilarity(fp, lig_rep_fps[cid]) >= 0.5:
            lig_clusters[cid].append(idx)
            lig_cluster_of[idx] = cid
            assigned = True
            break

    if not assigned:
        cid = f"C{cluster_id}"
        cluster_id += 1
        lig_clusters[cid] = [idx]
        lig_rep_fps[cid] = fp
        lig_cluster_of[idx] = cid

df['lig_cluster'] = df.index.map(lambda i: lig_cluster_of.get(i, f"lig_{i}"))
n_lig_clusters = df['lig_cluster'].nunique()
print(f"   配体相似度聚类数: {n_lig_clusters}")

gss_lig = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, lig_test = next(gss_lig.split(df, groups=df['lig_cluster']))
splits['Scaffold'] = lig_test
print(f"   测试集: {len(lig_test)}")

# ==================== 3. 蛋白序列相似度划分 ====================
print("3/4 蛋白序列提取与k-mer聚类...")

def extract_seq(pdb):
    protein_pdb = os.path.join(refined_dir, pdb, f"{pdb}_protein.pdb")
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb, protein_pdb)
    except:
        return ""
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    try:
                        seq += seq1(residue.get_resname())
                    except:
                        seq += 'X'
    return seq

# 检查是否已经提取过序列
if 'protein_seq' not in df.columns:
    print("   提取蛋白序列...")
    df['protein_seq'] = df['pdb'].apply(extract_seq)
    df.to_csv("pdbbind_valid.csv", index=False)

# ---- 纯 Python k-mer 聚类（替代 cd-hit） ----
def kmer_set(seq, k=3):
    """返回序列的 k-mer 集合"""
    if len(seq) < k:
        return set()
    return set(seq[i:i+k] for i in range(len(seq) - k + 1))

def jaccard(set1, set2):
    """Jaccard 相似度"""
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def greedy_kmer_cluster(seqs, k=3, threshold=0.3):
    """
    CD-HIT 风格的贪心聚类：
    - 按长度降序排列
    - 每条序列与已有簇代表比较 k-mer Jaccard
    - 相似度 >= threshold 则归入该簇，否则新建簇
    """
    # 按序列长度降序
    sorted_idx = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    kmer_sets = [kmer_set(seqs[i], k) for i in sorted_idx]

    clusters = {}       # cluster_id -> [original_indices]
    rep_kmers = {}      # cluster_id -> kmer set of representative
    cluster_of = {}     # original_index -> cluster_id

    cluster_id = 0
    for pos, (orig_idx, ks) in enumerate(zip(sorted_idx, kmer_sets)):
        if len(ks) == 0:
            cluster_of[orig_idx] = f"short_{orig_idx}"  # 序列太短，独立成簇
            continue

        assigned = False
        for cid in list(clusters.keys()):
            if jaccard(ks, rep_kmers[cid]) >= threshold:
                clusters[cid].append(orig_idx)
                cluster_of[orig_idx] = cid
                assigned = True
                break

        if not assigned:
            cid = f"C{cluster_id}"
            cluster_id += 1
            clusters[cid] = [orig_idx]
            rep_kmers[cid] = ks
            cluster_of[orig_idx] = cid

    return cluster_of

print("   运行 k-mer 聚类 (k=3, threshold=0.3)...")
seq_list = df['protein_seq'].tolist()
cluster_of = greedy_kmer_cluster(seq_list, k=3, threshold=0.3)
df['seq_cluster'] = df.index.map(lambda i: cluster_of.get(i, f"uniq_{i}"))

n_clusters = df['seq_cluster'].nunique()
print(f"   k-mer 聚类数: {n_clusters}")

gss_seq = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, seq_test = next(gss_seq.split(df, groups=df['seq_cluster']))
splits['Seq'] = seq_test
print(f"   测试集: {len(seq_test)}")

# ==================== 4. 结合模式划分 ====================
print("4/4 结合模式划分...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['binding_mode'] = kmeans.fit_predict(X)
mode_counts = df['binding_mode'].value_counts().to_dict()
print(f"   各模式样本数: {mode_counts}")

gss_mode = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, mode_test = next(gss_mode.split(df, groups=df['binding_mode']))
splits['Binding Mode'] = mode_test
print(f"   测试集: {len(mode_test)}")

# ==================== 保存 ====================
with open('splits.pkl', 'wb') as f:
    pickle.dump(splits, f)

print("\n========== 划分汇总 ==========")
for name, test_idx in splits.items():
    print(f"  {name}: 测试集 {len(test_idx)} 个")
print("\nsplits.pkl 已保存")
