import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DATA = r'D:\2026-repo\data\fan_vote_results_final.csv'
OUT_DIR = r'D:\2026-repo\figs'

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

BLUE = '#2b6cb0'
RED = '#c53030'
GRID = '#d9d9d9'
DIAG = '#bfbfbf'


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#4d4d4d')
        spine.set_linewidth(0.8)


# Figure 1: Rank vs Rank
rank_df = df[df['regime'] == 'rank'].copy()
rank_df['judge_rank'] = rank_df.groupby(['season', 'week'])['judge_total'].rank(ascending=False, method='first')
rank_df['fan_rank'] = rank_df.groupby(['season', 'week'])['fan_vote_share'].rank(ascending=False, method='first')

surv = rank_df[rank_df['eliminated_this_week'] == False]
elim = rank_df[rank_df['eliminated_this_week'] == True]

fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(surv['judge_rank'], surv['fan_rank'], s=36, c=BLUE, marker='o',
           alpha=0.85, edgecolors='white', linewidths=0.4, label='Survived')
ax.scatter(elim['judge_rank'], elim['fan_rank'], s=56, c=RED, marker='x',
           linewidths=1.8, label='Eliminated')

max_rank = int(max(rank_df['judge_rank'].max(), rank_df['fan_rank'].max()))
ax.plot([0.5, max_rank + 0.5], [0.5, max_rank + 0.5], linestyle='--', color=DIAG, linewidth=1)

ax.set_xlim(0.5, max_rank + 0.5)
ax.set_ylim(0.5, max_rank + 0.5)
ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xlabel("Judges' Score Rank (1 = best)")
ax.set_ylabel('Fans\' Vote Rank (1 = best)')
ax.set_title('Rank vs Rank: Judges vs Fans')
style_axes(ax)

ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

out1 = os.path.join(OUT_DIR, 'rank_rank_scatter.png')
fig.savefig(out1, dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)


# Figure 2: Percent vs Percent
pct_df = df[df['regime'] == 'percent'].copy()
pct_df['judge_share'] = pct_df.groupby(['season', 'week'])['judge_total'].transform(lambda s: s / s.sum())
pct_df['fan_share'] = pct_df['fan_vote_share']

surv = pct_df[pct_df['eliminated_this_week'] == False]
elim = pct_df[pct_df['eliminated_this_week'] == True]

fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(surv['judge_share'], surv['fan_share'], s=36, c=BLUE, marker='o',
           alpha=0.85, edgecolors='white', linewidths=0.4, label='Survived')
ax.scatter(elim['judge_share'], elim['fan_share'], s=56, c=RED, marker='x',
           linewidths=1.8, label='Eliminated')

ax.plot([0, 1], [0, 1], linestyle='--', color=DIAG, linewidth=1)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

ax.set_xlabel("Judges' Share (%)")
ax.set_ylabel('Fans\' Share (%)')
ax.set_title("Percent vs Percent: Judges' Share vs Fans' Share")
style_axes(ax)

ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

out2 = os.path.join(OUT_DIR, 'percent_percent_scatter.png')
fig.savefig(out2, dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

print(out1)
print(out2)
