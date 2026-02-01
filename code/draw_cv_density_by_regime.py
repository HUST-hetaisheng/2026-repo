"""
方案3：CV �?CI-Width 密度曲线（按 Regime 分层�?
验证：不确定性在不同参赛者之间是否保持一�?
- 上图：CV 密度曲线
- 下图：CI-Width 密度曲线
- 三条叠加的密度曲线（Rank=蓝，Percent=绿，Bottom-2=橙）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
df = pd.read_csv(DATA_DIR / "fan_vote_results_final.csv")

print(f"总记录数: {len(df)}")
print(f"CV 范围: [{df['cv'].min():.4f}, {df['cv'].max():.4f}]")
print(f"CI-Width 范围: [{df['ci_width'].min():.4f}, {df['ci_width'].max():.4f}]")
print(f"\n�?Regime 统计 (CV):")
print(df.groupby('regime')['cv'].agg(['count', 'mean', 'std', 'min', 'max']))
print(f"\n�?Regime 统计 (CI-Width):")
print(df.groupby('regime')['ci_width'].agg(['count', 'mean', 'std', 'min', 'max']))

# 设置样式
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 14,
})

# 定义颜色方案
colors = {
    'rank': '#3498db',      # 蓝色
    'percent': '#27ae60',   # 绿色
    'bottom2': '#e67e22'    # 橙色
}

labels = {
    'rank': 'Rank Regime (S1-2)',
    'percent': 'Percent Regime (S3-27)',
    'bottom2': 'Bottom-2 Regime (S28-34)'
}

# 创建 2x1 子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Distribution of Inference Uncertainty Across Contestants\n(Density by Voting Regime)", 
             fontsize=15, fontweight='bold', y=0.98)

# ========== 上图：CV 密度曲线 ==========
for regime in ['rank', 'percent', 'bottom2']:
    data = df[df['regime'] == regime]['cv'].dropna()
    
    if len(data) > 10:
        kde = stats.gaussian_kde(data, bw_method='scott')
        x_range = np.linspace(0, min(data.max() * 1.2, 1.0), 500)
        density = kde(x_range)
        
        ax1.fill_between(x_range, density, alpha=0.3, color=colors[regime])
        ax1.plot(x_range, density, color=colors[regime], linewidth=2.5, 
                label=f"{labels[regime]} (n={len(data)}, μ={data.mean():.3f})")
        ax1.axvline(data.mean(), color=colors[regime], linestyle='--', linewidth=1.5, alpha=0.7)

ax1.set_xlabel("Coefficient of Variation (CV)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Probability Density", fontsize=12, fontweight='bold')
ax1.set_title("(a) CV Distribution: Relative Uncertainty", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlim(0, 0.8)
ax1.set_ylim(0, None)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle=':')

# ========== 下图：CI-Width 密度曲线 ==========
for regime in ['rank', 'percent', 'bottom2']:
    data = df[df['regime'] == regime]['ci_width'].dropna()
    
    if len(data) > 10:
        kde = stats.gaussian_kde(data, bw_method='scott')
        x_range = np.linspace(0, min(data.max() * 1.2, 0.5), 500)
        density = kde(x_range)
        
        ax2.fill_between(x_range, density, alpha=0.3, color=colors[regime])
        ax2.plot(x_range, density, color=colors[regime], linewidth=2.5, 
                label=f"{labels[regime]} (n={len(data)}, μ={data.mean():.3f})")
        ax2.axvline(data.mean(), color=colors[regime], linestyle='--', linewidth=1.5, alpha=0.7)

ax2.set_xlabel("95% Confidence Interval Width", fontsize=12, fontweight='bold')
ax2.set_ylabel("Probability Density", fontsize=12, fontweight='bold')
ax2.set_title("(b) CI-Width Distribution: Absolute Uncertainty", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlim(0, 0.35)
ax2.set_ylim(0, None)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle=':')

# 添加注释�?
textstr = '\n'.join([
    'Key Findings:',
    '�?Percent regime: Tight clustering �?High consistency across contestants',
    '�?Bottom-2 regime: Wide spread �?Contestant-dependent uncertainty',
    '�?Rank regime: Intermediate spread �?Moderate consistency'
])
props = dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.95)
fig.text(0.5, 0.02, textstr, ha='center', fontsize=10, bbox=props)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# 保存图片
output_path = FIG_DIR / "cv_density_by_regime.png"
plt.savefig(output_path, dpi=800, bbox_inches='tight', facecolor='white')
print(f"\n已保�? {output_path}")
print(f"\n已保�? {output_path}")

plt.show()

# 额外统计：计算各 Regime 内的 CV 一致�?
print("\n" + "="*60)
print("参赛者间不确定性一致性分�?)
print("="*60)

for regime in ['rank', 'percent', 'bottom2']:
    data = df[df['regime'] == regime]['cv'].dropna()
    cv_of_cv = data.std() / data.mean() if data.mean() > 0 else 0
    
    print(f"\n{labels[regime]}:")
    print(f"  - 参赛者数: {len(data)}")
    print(f"  - CV均�? {data.mean():.4f}")
    print(f"  - CV标准�? {data.std():.4f}")
    print(f"  - CV的变异系�?(一致性指�?: {cv_of_cv:.4f}")
    print(f"  - 四分位距 (IQR): {data.quantile(0.75) - data.quantile(0.25):.4f}")
