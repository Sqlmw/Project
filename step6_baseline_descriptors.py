"""
分子描述符基线：用简单配体描述符作为特征，在相同划分策略上评估。
与PLIF对比，判断泛化衰减的瓶颈在特征端还是数据划分本身。
"""
import numpy as np
import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import TanimotoSimilarity
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

# ============= 加载数据 =============
df = pd.read_csv("pdbbind_valid.csv")
y = np.load("y_labels.npy")
refined_dir = "refined-set"
indices = np.arange(len(y))

# ============= 计算配体描述符 =============
print("计算配体描述符...")

DESC_NAMES = [
    'MolWt', 'LogP', 'NumHAcceptors', 'NumHDonors',
    'NumRotatableBonds', 'NumRings', 'NumAromaticRings', 'TPSA',
    'FractionCsp3', 'NumHeteroatoms', 'HeavyAtomCount',
]

def compute_descriptors(pdb, lig_suffix):
    lig_path = os.path.join(refined_dir, pdb, f"{pdb}_ligand.{lig_suffix}")
    try:
        if lig_suffix == 'mol2':
            mol = Chem.MolFromMol2File(lig_path, sanitize=True, removeHs=True)
        else:
            mol = Chem.MolFromMolFile(lig_path, sanitize=True, removeHs=True)
        if mol is None: return None
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumRings': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        }
    except: return None

X_desc_list = []
for _, row in df.iterrows():
    d = compute_descriptors(row['pdb'], row['ligand_suffix'])
    if d is not None:
        X_desc_list.append([d[name] for name in DESC_NAMES])
    else:
        X_desc_list.append([np.nan] * len(DESC_NAMES))

X_desc = np.array(X_desc_list, dtype=float)
for j in range(X_desc.shape[1]):
    col = X_desc[:, j]
    mask = np.isnan(col)
    if mask.any():
        col[mask] = np.nanmean(col[~mask])
        X_desc[:, j] = col

print(f"  维度: {X_desc.shape[1]}")

# ============= 预计算划分配套 =============
print("预计算划分配套...")

fps = {}
for idx, row in df.iterrows():
    lig_path = os.path.join(refined_dir, row['pdb'],
                            f"{row['pdb']}_ligand.{row['ligand_suffix']}")
    try:
        if row['ligand_suffix'] == 'mol2':
            mol = Chem.MolFromMol2File(lig_path, sanitize=True, removeHs=True)
        else:
            mol = Chem.MolFromMolFile(lig_path, sanitize=True, removeHs=True)
        if mol is not None:
            fps[idx] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except: pass

lig_clusters, lig_rep, lig_of = {}, {}, {}
cid = 0
for idx in list(fps.keys()):
    fp = fps[idx]; assigned = False
    for c in list(lig_clusters.keys()):
        if TanimotoSimilarity(fp, lig_rep[c]) >= 0.5:
            lig_clusters[c].append(idx); lig_of[idx] = c; assigned = True; break
    if not assigned:
        lig_clusters[f'C{cid}'] = [idx]; lig_rep[f'C{cid}'] = fp
        lig_of[idx] = f'C{cid}'; cid += 1
lig_groups = df.index.map(lambda i: lig_of.get(i, f'lig_{i}'))

def extract_seq(pdb):
    protein_pdb = os.path.join(refined_dir, pdb, f"{pdb}_protein.pdb")
    parser = PDBParser(QUIET=True)
    try: structure = parser.get_structure(pdb, protein_pdb)
    except: return ""
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    try: seq += seq1(residue.get_resname())
                    except: seq += 'X'
    return seq

if 'protein_seq' not in df.columns:
    df['protein_seq'] = df['pdb'].apply(extract_seq)

def kmer_set(seq, k=3):
    if len(seq) < k: return set()
    return set(seq[i:i+k] for i in range(len(seq)-k+1))

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    inter = len(s1 & s2); union = len(s1 | s2)
    return inter/union if union>0 else 0.0

def greedy_kmer_cluster(seqs, k=3, threshold=0.3):
    sorted_idx = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    kmers = [kmer_set(seqs[i], k) for i in sorted_idx]
    clusters, reps, co = {}, {}, {}
    cid_cnt = 0
    for orig_idx, ks in zip(sorted_idx, kmers):
        if len(ks)==0: co[orig_idx]=f'short_{orig_idx}'; continue
        assigned = False
        for ci in list(clusters.keys()):
            if jaccard(ks, reps[ci]) >= threshold:
                clusters[ci].append(orig_idx); co[orig_idx]=ci; assigned=True; break
        if not assigned:
            clusters[f'C{cid_cnt}']=[orig_idx]; reps[f'C{cid_cnt}']=ks
            co[orig_idx]=f'C{cid_cnt}'; cid_cnt+=1
    return co

seq_cluster_of = greedy_kmer_cluster(df['protein_seq'].tolist(), k=3, threshold=0.3)
seq_groups = df.index.map(lambda i: seq_cluster_of.get(i, f'uniq_{i}'))

# ============= 评估 =============
N_REPEATS = 5
print(f"\n========== 描述符基线 ({N_REPEATS} 次重复) ==========")

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    rho, _ = spearmanr(y_test, pred)
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test)**2))
    return rho, mae, rmse

all_results = {n: {'Spearman':[], 'MAE':[], 'RMSE':[]}
               for n in ['Random','Scaffold','Seq','Binding Mode']}

for seed in range(N_REPEATS):
    splits = {}
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    splits['Random'] = test_idx

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=lig_groups))
    splits['Scaffold'] = test_idx

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=seq_groups))
    splits['Seq'] = test_idx

    bm_labels = KMeans(n_clusters=5, random_state=seed, n_init=10).fit_predict(X_desc)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, test_idx = next(gss.split(df, groups=bm_labels))
    splits['Binding Mode'] = test_idx

    for name, test_idx in splits.items():
        train_idx = np.setdiff1d(indices, test_idx)
        rf = RandomForestRegressor(n_estimators=400, max_depth=15,
                                   min_samples_leaf=3, min_samples_split=6,
                                   max_features='sqrt', n_jobs=-1, random_state=seed)
        rf.fit(X_desc[train_idx], y[train_idx])
        rho, mae, rmse = evaluate(rf, X_desc[test_idx], y[test_idx])
        all_results[name]['Spearman'].append(rho)
        all_results[name]['MAE'].append(mae)
        all_results[name]['RMSE'].append(rmse)
    print(f"  seed={seed} 完成")

# ============= PLIF vs 描述符 对比 =============
with open('results.json') as f:
    plif = json.load(f)

print(f"\n========== PLIF vs 描述符 ==========")
print(f"{'Split':<15} {'PLIF':>18} {'Descriptors':>18}  {'Delta':>8}")
print("-" * 65)
for name in ['Random','Scaffold','Seq','Binding Mode']:
    r = all_results[name]
    p_mean = plif[name]['Spearman_mean']
    d_mean = np.mean(r['Spearman'])
    delta = p_mean - d_mean
    p_str = f"{p_mean:.3f}+-{plif[name]['Spearman_std']:.3f}"
    d_str = f"{d_mean:.3f}+-{np.std(r['Spearman']):.3f}"
    sign = '+' if delta > 0 else ''
    print(f"{name:<15} {p_str:>18} {d_str:>18}  {sign}{delta:.3f}")

# ============= 泛化衰减对比 =============
print(f"\n{'':<15} {'PLIF drop':>18} {'Descriptors drop':>18}")
print("-" * 55)
for name in ['Scaffold','Seq','Binding Mode']:
    pr = plif['Random']['Spearman_mean']
    dr = np.mean(all_results['Random']['Spearman'])
    p_drop = (pr - plif[name]['Spearman_mean']) / pr * 100
    d_drop = (dr - np.mean(all_results[name]['Spearman'])) / dr * 100
    print(f"{name:<15} {'-'+str(round(p_drop))+'%':>18} {'-'+str(round(d_drop))+'%':>18}")

# ============= 可视化 =============
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

split_names = ['Random', 'Scaffold', 'Seq', 'Binding Mode']
p_means = [plif[n]['Spearman_mean'] for n in split_names]
p_stds = [plif[n]['Spearman_std'] for n in split_names]
d_means = [np.mean(all_results[n]['Spearman']) for n in split_names]
d_stds = [np.std(all_results[n]['Spearman']) for n in split_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 左侧：绝对性能对比
x = np.arange(len(split_names))
width = 0.35
ax1.bar(x - width/2, p_means, width, yerr=p_stds, capsize=4,
        color='#2E86AB', label='PLIF (5-dim)')
ax1.bar(x + width/2, d_means, width, yerr=d_stds, capsize=4,
        color='#A23B72', label='Descriptors (11-dim)')
ax1.set_xticks(x)
ax1.set_xticklabels(split_names, fontsize=11)
ax1.set_ylabel('Spearman rho', fontsize=12)
ax1.set_title('Absolute Performance', fontsize=13)
ax1.legend(loc='upper right', frameon=False, fontsize=10)
ax1.set_ylim(0, 0.85)

# 右侧：泛化衰减对比
p_drops = [(p_means[0] - p_means[i]) / p_means[0] * 100 for i in range(1, 4)]
d_drops = [(d_means[0] - d_means[i]) / d_means[0] * 100 for i in range(1, 4)]
drop_names = split_names[1:]

x2 = np.arange(len(drop_names))
ax2.bar(x2 - width/2, p_drops, width, color='#2E86AB', label='PLIF')
ax2.bar(x2 + width/2, d_drops, width, color='#A23B72', label='Descriptors')
ax2.set_xticks(x2)
ax2.set_xticklabels(drop_names, fontsize=11)
ax2.set_ylabel('Drop from Random (%)', fontsize=12)
ax2.set_title('Drop Relative to Random', fontsize=13)
ax2.legend(loc='upper right', frameon=False, fontsize=10)
ax2.set_ylim(0, 65)

plt.suptitle('PLIF vs Molecular Descriptors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('baseline_comparison_tuned.png', dpi=300)
print("\n已保存: baseline_comparison_tuned.png")

# 保存
desc_results = {}
for name in ['Random','Scaffold','Seq','Binding Mode']:
    r = all_results[name]
    desc_results[name] = {
        'Spearman_mean': float(np.mean(r['Spearman'])),
        'Spearman_std': float(np.std(r['Spearman'])),
        'MAE_mean': float(np.mean(r['MAE'])),
        'MAE_std': float(np.std(r['MAE'])),
        'RMSE_mean': float(np.mean(r['RMSE'])),
        'RMSE_std': float(np.std(r['RMSE'])),
    }
with open('baseline_descriptors_tuned.json', 'w') as f:
    json.dump(desc_results, f, indent=2)

print("\n已保存: baseline_descriptors.json")
