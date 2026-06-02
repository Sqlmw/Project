"""
第5天：泛化衰减热图
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

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

print("\n========== 第5天完成 ==========")
print("生成文件: heatmap.png, barplot.png")
