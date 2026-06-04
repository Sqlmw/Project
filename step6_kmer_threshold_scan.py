"""
k-mer Jaccard 阈值扫描：评估不同阈值下Seq划分的泛化难度
"""
import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from scipy.stats import spearmanr
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# ============= 加载数据 =============
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
df = pd.read_csv("pdbbind_valid.csv")
refined_dir = "refined-set"
indices = np.arange(len(X))

# ============= 提取蛋白序列 =============
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
    print("提取蛋白序列...")
    df['protein_seq'] = df['pdb'].apply(extract_seq)

# ============= k-mer 聚类 =============
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
    cid = 0
    for orig_idx, ks in zip(sorted_idx, kmer_sets):
        if len(ks) == 0:
            cluster_of[orig_idx] = f"short_{orig_idx}"
            continue
        assigned = False
        for ci in list(clusters.keys()):
            if jaccard(ks, rep_kmers[ci]) >= threshold:
                clusters[ci].append(orig_idx)
                cluster_of[orig_idx] = ci
                assigned = True
                break
        if not assigned:
            clusters[f"C{cid}"] = [orig_idx]
            rep_kmers[f"C{cid}"] = ks
            cluster_of[orig_idx] = f"C{cid}"
            cid += 1
    return cluster_of

# ============= 扫描阈值 =============
print("扫描 k-mer Jaccard 阈值 0.2-0.6...")
seq_list = df['protein_seq'].tolist()
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
results = {}

for th in thresholds:
    cluster_of = greedy_kmer_cluster(seq_list, k=3, threshold=th)
    seq_groups = df.index.map(lambda i: cluster_of.get(i, f"uniq_{i}"))
    n_clusters = seq_groups.nunique()

    s_vals, m_vals, r_vals = [], [], []
    for seed in range(5):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        _, test_idx = next(gss.split(df, groups=seq_groups))
        test_n = len(test_idx)
        train_idx = np.setdiff1d(indices, test_idx)

        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        rho, _ = spearmanr(y[test_idx], pred)
        s_vals.append(rho)
        m_vals.append(np.mean(np.abs(pred - y[test_idx])))
        r_vals.append(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))

    results[th] = {
        'clusters': n_clusters,
        'test_n': test_n,
        'spearman_mean': float(np.mean(s_vals)),
        'spearman_std': float(np.std(s_vals)),
        'mae_mean': float(np.mean(m_vals)),
        'rmse_mean': float(np.mean(r_vals)),
    }
    print(f"  th={th}: clusters={n_clusters}, Spearman={np.mean(s_vals):.3f}+-{np.std(s_vals):.3f}")

# Random baseline
s_vals = []
for seed in range(5):
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx = np.setdiff1d(indices, test_idx)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    rf.fit(X[train_idx], y[train_idx])
    pred = rf.predict(X[test_idx])
    rho, _ = spearmanr(y[test_idx], pred)
    s_vals.append(rho)
random_baseline = float(np.mean(s_vals))
print(f"\nRandom baseline: {random_baseline:.3f}")

# ============= 可视化 =============
fig, ax1 = plt.subplots(figsize=(8, 5))

spearman_mean = [results[t]['spearman_mean'] for t in thresholds]
spearman_std = [results[t]['spearman_std'] for t in thresholds]
clusters = [results[t]['clusters'] for t in thresholds]

color = '#2E86AB'
ax1.errorbar(thresholds, spearman_mean, yerr=spearman_std, color=color,
             marker='o', linewidth=2, markersize=8, capsize=5, label='Seq split')
ax1.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1.5,
            label=f'Random baseline ({random_baseline:.3f})')
ax1.set_xlabel('Jaccard threshold', fontsize=12)
ax1.set_ylabel('Spearman rho', fontsize=12, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.2, 0.6)

ax2 = ax1.twinx()
color2 = '#A23B72'
ax2.plot(thresholds, clusters, color=color2, marker='s', linewidth=2,
         markersize=8, linestyle='--', label='Clusters')
ax2.set_ylabel('Number of clusters', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)

plt.title('Effect of k-mer Jaccard Threshold on Seq Split', fontsize=13)
plt.tight_layout()
plt.savefig('kmer_threshold_scan.png', dpi=300)

# 保存数据
with open('kmer_scan_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n已保存: kmer_threshold_scan.png, kmer_scan_results.json")
