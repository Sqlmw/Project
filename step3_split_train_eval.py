"""
第4天：对每种划分训练RF，多次重复评估，报告均值±标准差
"""
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.cluster import KMeans
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

N_REPEATS = 5
print(f"总样本: {len(X)}, 特征: {X.shape[1]}维, pK: [{y.min():.2f}, {y.max():.2f}]")
print(f"重复次数: {N_REPEATS}")

# ============= 预计算（不随种子变化） =============

# --- 配体 Morgan 指纹 ---
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

# Tanimoto 贪心聚类
lig_cluster_of = {}
lig_clusters = {}
lig_rep_fps = {}
cluster_id = 0
valid_indices_fp = list(fps.keys())
for idx in valid_indices_fp:
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

# --- 蛋白序列 + k-mer 聚类 ---
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
    clusters = {}
    rep_kmers = {}
    cluster_of = {}
    cluster_id_cnt = 0
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
            cid = f"C{cluster_id_cnt}"
            cluster_id_cnt += 1
            clusters[cid] = [orig_idx]
            rep_kmers[cid] = ks
            cluster_of[orig_idx] = cid
    return cluster_of

seq_list = df['protein_seq'].tolist()
seq_cluster_of = greedy_kmer_cluster(seq_list, k=3, threshold=0.3)
seq_groups = df.index.map(lambda i: seq_cluster_of.get(i, f"uniq_{i}"))


# ============= 多次重复评估 =============
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    rho, _ = spearmanr(y_test, pred)
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    return {'Spearman': rho, 'MAE': mae, 'RMSE': rmse}

# 收集所有重复的结果
all_results = {name: {'Spearman': [], 'MAE': [], 'RMSE': []}
               for name in ['Random', 'Scaffold', 'Seq', 'Binding Mode']}

print(f"\n========== 运行 {N_REPEATS} 次重复评估 ==========")

for seed in range(N_REPEATS):
    splits = {}

    # 1. Random
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    splits['Random'] = test_idx

    # 2. Scaffold (Morgan + Tanimoto)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=lig_groups))
    splits['Scaffold'] = test_idx

    # 3. Seq (k-mer)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=seq_groups))
    splits['Seq'] = test_idx

    # 4. Binding Mode (KMeans with seed)
    kmeans = KMeans(n_clusters=5, random_state=seed, n_init=10)
    bm_labels = kmeans.fit_predict(X)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=bm_labels))
    splits['Binding Mode'] = test_idx

    # 训练+评估
    for name, test_idx in splits.items():
        train_idx = np.setdiff1d(indices, test_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

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
    spearman_str = f"{np.mean(r['Spearman']):.3f}±{np.std(r['Spearman']):.3f}"
    mae_str = f"{np.mean(r['MAE']):.3f}±{np.std(r['MAE']):.3f}"
    rmse_str = f"{np.mean(r['RMSE']):.3f}±{np.std(r['RMSE']):.3f}"
    print(f"{name:<15} {spearman_str:>18} {mae_str:>16} {rmse_str:>16}")
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
