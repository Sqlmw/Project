"""
第5天：泛化衰减热图 + 配体应变能分析与修正
"""
import json
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)

# ============ Part 1: 热图 ============
print("Part 1: 泛化衰减热图...")
with open('results.json') as f:
    results = json.load(f)

# 提取 Spearman rho
data_for_heatmap = {}
for split_name, models in results.items():
    data_for_heatmap[split_name] = {
        'RF': models['RF']['Spearman'],
        'MLP': models['MLP']['Spearman']
    }

df_plot = pd.DataFrame(data_for_heatmap).T
print(df_plot)

plt.figure(figsize=(8, 5))
sns.heatmap(df_plot.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.3, vmax=0.95, linewidths=1)
plt.title('Generalization Performance Across Data Splits (Spearman rho)')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
print("  已保存 heatmap.png")

# 柱状图版本（更直观）
plt.figure(figsize=(10, 5))
x = np.arange(len(df_plot))
width = 0.35
plt.bar(x - width/2, df_plot['RF'], width, label='Random Forest', color='#2E86AB')
plt.bar(x + width/2, df_plot['MLP'], width, label='MLP', color='#A23B72')
plt.xticks(x, df_plot.index)
plt.ylabel('Spearman rho')
plt.title('Model Performance Across Data Splits')
plt.legend()
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('barplot.png', dpi=300)
print("  已保存 barplot.png")

# ============ Part 2: 配体复杂度与预测误差分析 ============
print("\nPart 2: 配体复杂度与预测误差分析...")

df = pd.read_csv("pdbbind_valid.csv")
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
refined_dir = "refined-set"

def compute_ligand_props(pdb, lig_suffix):
    """计算配体复杂度指标：可旋转键数、重原子数、芳香环数"""
    lig_path = f"{refined_dir}/{pdb}/{pdb}_ligand.{lig_suffix}"
    try:
        if lig_suffix == 'mol2':
            mol = Chem.MolFromMol2File(lig_path, removeHs=True)
        else:
            mol = Chem.MolFromMolFile(lig_path, removeHs=True)
        if mol is None:
            return None
        n_rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        n_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        n_arom_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
        # 复杂度分数 = 可旋转键 / 重原子（归一化的柔性度量）
        flex_score = n_rot / max(n_heavy, 1)
        return {'n_rot': n_rot, 'n_heavy': n_heavy, 'n_rings': n_rings,
                'n_arom_rings': n_arom_rings, 'flex_score': flex_score}
    except:
        return None

print("  计算配体复杂度...")
props = df.apply(lambda r: compute_ligand_props(r['pdb'], r['ligand_suffix']), axis=1)
valid_mask = ~props.isna()
print(f"  有效样本: {valid_mask.sum()}/{len(df)}")

# 提取属性
df['n_rot'] = props.apply(lambda p: p['n_rot'] if p else None)
df['n_heavy'] = props.apply(lambda p: p['n_heavy'] if p else None)
df['flex_score'] = props.apply(lambda p: p['flex_score'] if p else None)

n_rot = df['n_rot'].values
n_heavy = df['n_heavy'].values
flex_score = df['flex_score'].values

X_valid = X[valid_mask]
y_valid = y[valid_mask]
n_rot_valid = n_rot[valid_mask]
n_heavy_valid = n_heavy[valid_mask]
flex_valid = flex_score[valid_mask]

# 用随机划分训练模型
indices_valid = np.arange(len(X_valid))
train_idx, test_idx = train_test_split(indices_valid, test_size=0.2, random_state=42)

X_train, y_train = X_valid[train_idx], y_valid[train_idx]
X_test, y_test = X_valid[test_idx], y_valid[test_idx]
rot_test = n_rot_valid[test_idx]
rot_train = n_rot_valid[train_idx]
heavy_test = n_heavy_valid[test_idx]
flex_test = flex_valid[test_idx]

print(f"\n  训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 训练 RF
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
pred_test = rf.predict(X_test)
residuals_test = np.abs(pred_test - y_test)
pred_train = rf.predict(X_train)
residuals_train = np.abs(pred_train - y_train)

# 残差 vs 可旋转键数
corr_rot, p_rot = spearmanr(residuals_test, rot_test)
corr_heavy, p_heavy = spearmanr(residuals_test, heavy_test)
corr_flex, p_flex = spearmanr(residuals_test, flex_test)
print(f"  残差-可旋转键 Spearman: {corr_rot:.3f} (p={p_rot:.3f})")
print(f"  残差-重原子数 Spearman: {corr_heavy:.3f} (p={p_heavy:.3f})")
print(f"  残差-柔性分数 Spearman: {corr_flex:.3f} (p={p_flex:.3f})")

# 使用相关性最强的指标
strain_proxy = rot_test
strain_proxy_train = rot_train
strain_label = 'Number of Rotatable Bonds'
strain_fname = 'nrot'

# 散点图
plt.figure(figsize=(7, 5))
plt.scatter(strain_proxy, residuals_test, alpha=0.4, c='#2E86AB', edgecolors='none')
plt.xlabel(strain_label)
plt.ylabel('Absolute Prediction Error (pK)')
plt.title(f'Error vs {strain_label} (Spearman rho = {corr_rot:.3f})')
z = np.polyfit(strain_proxy, residuals_test, 1)
p_line = np.poly1d(z)
x_line = np.linspace(strain_proxy.min(), strain_proxy.max(), 100)
plt.plot(x_line, p_line(x_line), 'r--', linewidth=2, label='Trend')
plt.legend()
plt.tight_layout()
plt.savefig('strain_residual.png', dpi=300)
print("  已保存 strain_residual.png")

# 修正模型
corr_model = LinearRegression()
corr_model.fit(strain_proxy_train.reshape(-1, 1), residuals_train)
print(f"  修正系数: {corr_model.coef_[0]:.4f} pK/rotatable_bond")

# 修正预测
strain_correction = corr_model.predict(strain_proxy.reshape(-1, 1))
pred_corrected = pred_test - strain_correction

rho_before, _ = spearmanr(y_test, pred_test)
rho_after, _ = spearmanr(y_test, pred_corrected)
print(f"\n  修正前 Spearman: {rho_before:.3f}")
print(f"  修正后 Spearman: {rho_after:.3f}")
print(f"  改进: {rho_after - rho_before:+.3f}")

# 修正前后对比图
plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred_test, alpha=0.3, label=f'Original (rho={rho_before:.3f})', c='#2E86AB')
plt.scatter(y_test, pred_corrected, alpha=0.3, label=f'Corrected (rho={rho_after:.3f})', c='#A23B72')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.5)
plt.xlabel('True pK')
plt.ylabel('Predicted pK')
plt.title('Rotatable Bond Correction Effect')
plt.legend()
plt.tight_layout()
plt.savefig('correction_comparison.png', dpi=300)
print("  已保存 correction_comparison.png")

# 柔性分数的直方图
plt.figure(figsize=(7, 4))
plt.hist(flex_test, bins=30, color='#2E86AB', alpha=0.7, edgecolor='k')
plt.xlabel('Flexibility Score (N_rot / N_heavy)')
plt.ylabel('Count')
plt.title('Distribution of Ligand Flexibility')
plt.tight_layout()
plt.savefig('flexibility_hist.png', dpi=300)

# ============ Part 3: 失败案例分析（饼图） ============
print("\nPart 3: 失败案例分析...")

error_df = pd.DataFrame({
    'pdb': df[valid_mask].iloc[test_idx]['pdb'].values,
    'residual': residuals_test,
    'n_rot': rot_test,
    'n_heavy': heavy_test,
    'true_pK': y_test,
    'pred_pK': pred_test
})
top_errors = error_df.nlargest(10, 'residual')
print("\nTop 10 最大误差样本:")
for i, row in top_errors.iterrows():
    print(f"  {row['pdb']}: true={row['true_pK']:.2f}, pred={row['pred_pK']:.2f}, "
          f"error={row['residual']:.2f}, rot_bonds={row['n_rot']:.0f}")

# 分类
high_rot = (rot_test > np.percentile(rot_test, 75))
high_error = (residuals_test > np.percentile(residuals_test, 75))

cat_high = np.sum(high_rot & high_error)
cat_other = np.sum(~high_rot & high_error)
cat_low = np.sum(~high_error)

categories = ['High Flexibility\n+ High Error', 'Other High Error', 'Low Error']
sizes = [cat_high, cat_other, cat_low]
colors = ['#E63946', '#F4A261', '#2A9D8F']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=(0.05, 0.02, 0))
plt.title('Error Source Distribution')
plt.tight_layout()
plt.savefig('error_pie.png', dpi=300)
print("  已保存 error_pie.png")

print("\n========== 第5天完成 ==========")
print("生成文件: heatmap.png, barplot.png, strain_residual.png, correction_comparison.png, error_pie.png, flexibility_hist.png")
