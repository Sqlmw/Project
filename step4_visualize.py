"""
Step 5：泛化衰减热图 + 误差棒柱状图
输入: results.json（step4 的评估结果）
输出: heatmap.png, barplot.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

with open('results.json') as f:
    results = json.load(f)

# 提取每个划分的 Spearman 均值和标准差
names = list(results.keys())
spearman_means = [results[n]['Spearman_mean'] for n in names]
spearman_stds = [results[n]['Spearman_std'] for n in names]

# 终端打印
print("Split            Spearman")
print("-" * 35)
for n, m, s in zip(names, spearman_means, spearman_stds):
    print(f"{n:<15} {m:.3f} +/- {s:.3f}")

# ============= 热图 =============
data = {n: {'RF': results[n]['Spearman_mean']} for n in names}
df_plot = pd.DataFrame(data).T

plt.figure(figsize=(5, 4))
# cmap 使用与柱状图同色系的蓝色渐变；vmin/vmax 缩到数据实际范围以增强对比
sns.heatmap(df_plot.astype(float), annot=True, fmt='.3f',
            cmap=sns.light_palette('#2E86AB', as_cmap=True),
            vmin=0.30, vmax=0.60, linewidths=1)
plt.title('Generalization Performance (Spearman rho)')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
print("  已保存 heatmap.png")

# ============= 柱状图（带误差棒） =============
plt.figure(figsize=(8, 5))
x = np.arange(len(names))
plt.bar(x, spearman_means, color='#2E86AB', yerr=spearman_stds,
        capsize=5, error_kw={'linewidth': 1.5})
plt.xticks(x, names)
plt.ylabel('Spearman rho')
plt.title('RF Performance Across Data Splits (mean +/- std)')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)  # 参考线
plt.tight_layout()
plt.savefig('barplot.png', dpi=300)
print("  已保存 barplot.png")

print("\n========== 第4天完成 ==========")
