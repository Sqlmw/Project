"""
Step 4：对每种划分独立训练 RF，5 次重复评估，报告均值±标准差
核心改进：每种划分用自己的训练集（互补集），避免信息泄漏
"""
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

# ============= 加载数据 =============
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
df = pd.read_csv("pdbbind_valid.csv")
refined_dir = "refined-set"
indices = np.arange(len(X))

N_REPEATS = 5  # 用 5 个不同随机种子重复评估
print(f"总样本: {len(X)}, 特征: {X.shape[1]}维, pK: [{y.min():.2f}, {y.max():.2f}]")
print(f"重复次数: {N_REPEATS}")

# ============= 预计算：配体 Morgan 指纹 + Tanimoto 聚类 =============
# 聚类是确定性的，算一次即可，5 次重复共用
print("\n预计算配体 Morgan 指纹...")

def get_morgan_fp(pdb, lig_suffix):
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

fps = {}
for idx, row in df.iterrows():
    fp = get_morgan_fp(row['pdb'], row['ligand_suffix'])
    if fp is not None:
        fps[idx] = fp

# Tanimoto 贪心聚类：≥ 0.5 归入同簇
lig_cluster_of, lig_clusters, lig_rep_fps = {}, {}, {}
cluster_id = 0
for idx in list(fps.keys()):
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

lig_groups = df.index.map(lambda i: lig_cluster_of.get(i, f"lig_{i}"))

# ============= 预计算：蛋白序列 + k-mer 聚类 =============
print("预计算蛋白序列 k-mer 聚类...")

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

if 'protein_seq' not in df.columns:
    df['protein_seq'] = df['pdb'].apply(extract_seq)

def kmer_set(seq, k=3):
    if len(seq) < k:
        return set()
    return set(seq[i:i+k] for i in range(len(seq) - k + 1))

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def greedy_kmer_cluster(seqs, k=3, threshold=0.3):
    sorted_idx = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    kmer_sets = [kmer_set(seqs[i], k) for i in sorted_idx]
    clusters, rep_kmers, cluster_of = {}, {}, {}
    cid_cnt = 0
    for orig_idx, ks in zip(sorted_idx, kmer_sets):
        if len(ks) == 0:
            cluster_of[orig_idx] = f"short_{orig_idx}"
            continue
        assigned = False
        for cid in list(clusters.keys()):
            if jaccard(ks, rep_kmers[cid]) >= threshold:
                clusters[cid].append(orig_idx)
                cluster_of[orig_idx] = cid
                assigned = True
                break
        if not assigned:
            cid = f"C{cid_cnt}"
            cid_cnt += 1
            clusters[cid] = [orig_idx]
            rep_kmers[cid] = ks
            cluster_of[orig_idx] = cid
    return cluster_of

seq_list = df['protein_seq'].tolist()
seq_cluster_of = greedy_kmer_cluster(seq_list, k=3, threshold=0.3)
seq_groups = df.index.map(lambda i: seq_cluster_of.get(i, f"uniq_{i}"))

# ============= 多次重复评估 =============
def evaluate(model, X_test, y_test):
    """计算三个回归评估指标"""
    pred = model.predict(X_test)
    rho, _ = spearmanr(y_test, pred)
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    return {'Spearman': rho, 'MAE': mae, 'RMSE': rmse}

# ============= 预计算：结合口袋残基组成分类（替代 KMeans） =============
print("\n预计算结合口袋类型...")

# 氨基酸分类
HYDROPHOBIC_AA = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO', 'TYR', 'CYS'}
CHARGED_AA = {'LYS', 'ARG', 'HIS', 'ASP', 'GLU'}
AROMATIC_AA = {'PHE', 'TYR', 'TRP', 'HIS'}
POLAR_AA = {'SER', 'THR', 'ASN', 'GLN'}

def classify_pocket(pdb):
    """读 pocket.pdb，按残基组成比例分类为 疏水型/带电型/极性型"""
    pocket_path = os.path.join(refined_dir, pdb, f"{pdb}_pocket.pdb")
    if not os.path.exists(pocket_path):
        return 0  # 缺失文件归默认类
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb, pocket_path)
    except:
        return 0
    # 统计各类残基数量（去重：同一残基多种原子只计一次）
    seen_residues = set()
    counts = {'hydrophobic': 0, 'charged': 0, 'aromatic': 0, 'polar': 0}
    for residue in structure.get_residues():
        resname = residue.get_resname().strip()
        res_id = (residue.get_full_id()[2], residue.get_full_id()[3][1])
        if res_id in seen_residues:
            continue
        seen_residues.add(res_id)
        if resname in HYDROPHOBIC_AA:
            counts['hydrophobic'] += 1
        if resname in CHARGED_AA:
            counts['charged'] += 1
        if resname in AROMATIC_AA:
            counts['aromatic'] += 1
        if resname in POLAR_AA:
            counts['polar'] += 1
    total = sum(counts.values())
    if total == 0:
        return 0
    # 哪类占比最高就归哪类
    if counts['charged'] / total >= 0.35:
        return 2  # 带电型口袋
    elif counts['hydrophobic'] / total >= 0.50:
        return 1  # 疏水型口袋
    else:
        return 0  # 混合/极性型口袋

# 预计算所有口袋类型（规则确定，不随种子变化）
df['pocket_type'] = df['pdb'].apply(classify_pocket)
pocket_type_counts = df['pocket_type'].value_counts().to_dict()
print(f"   口袋类型分布: {pocket_type_counts}")

pocket_groups = df['pocket_type'].values  # 作为 GroupShuffleSplit 的分组
all_results = {name: {'Spearman': [], 'MAE': [], 'RMSE': []}
               for name in ['Random', 'Scaffold', 'Seq', 'Binding Mode']}

print(f"\n========== 运行 {N_REPEATS} 次重复评估 ==========")

for seed in range(N_REPEATS):
    splits = {}

    # 1. Random：每次用不同 seed 生成新的随机切分
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    splits['Random'] = test_idx

    # 2. Scaffold：分组已固定，seed 改变的是哪些簇进测试集
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=lig_groups))
    splits['Scaffold'] = test_idx

    # 3. Seq：同上，按序列簇分组
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=seq_groups))
    splits['Seq'] = test_idx

    # 4. Binding Mode（口袋残基组成分类，规则确定，不依赖 PLIF/KMeans）
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=pocket_groups))
    splits['Binding Mode'] = test_idx

    # 对每种划分独立训练评估（互补集 = 全部 - 测试集）
    for name, test_idx in splits.items():
        train_idx = np.setdiff1d(indices, test_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 使用 step7 调优后的 RF 参数：限制树深、最小叶样本，防止过拟合
        # 调优前默认参数（用于对比）
        rf = RandomForestRegressor(n_estimators=400, max_depth=15,
                                   min_samples_leaf=3, min_samples_split=6,
                                   max_features='sqrt', n_jobs=-1, random_state=seed)
        rf.fit(X_train, y_train)
        res = evaluate(rf, X_test, y_test)
        for metric in ['Spearman', 'MAE', 'RMSE']:
            all_results[name][metric].append(res[metric])

    print(f"  seed={seed} 完成")

# ============= 汇总 =============
print(f"\n========== 汇总 (均值±标准差, {N_REPEATS} 次重复) ==========")
print(f"{'Split':<15} {'Spearman':>18} {'MAE':>16} {'RMSE':>16}")
print("-" * 68)

final_results = {}
for name in ['Random', 'Scaffold', 'Seq', 'Binding Mode']:
    r = all_results[name]
    print(f"{name:<15} {np.mean(r['Spearman']):.3f}±{np.std(r['Spearman']):.3f}"
          f"{'':>6} {np.mean(r['MAE']):.3f}±{np.std(r['MAE']):.3f}"
          f"{'':>6} {np.mean(r['RMSE']):.3f}±{np.std(r['RMSE']):.3f}")
    final_results[name] = {
        'Spearman_mean': float(np.mean(r['Spearman'])),
        'Spearman_std': float(np.std(r['Spearman'])),
        'MAE_mean': float(np.mean(r['MAE'])),
        'MAE_std': float(np.std(r['MAE'])),
        'RMSE_mean': float(np.mean(r['RMSE'])),
        'RMSE_std': float(np.std(r['RMSE'])),
    }

with open('results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n已保存: results.json")
