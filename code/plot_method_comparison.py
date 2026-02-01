# -*- coding: utf-8 -*-
"""
Elimination Boundary Comparison
Show how Rank vs Percentage methods have different elimination thresholds

Key insight:
- In Percentage method, fan vote share has ~2.8x higher variance than judge share
- This means fan votes have more "weight" in determining who is eliminated
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_FAN = Path(r"D:\2026-repo\data\fan_vote_results_final.csv")
OUT_DIR = Path(r"D:\2026-repo\figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": "#333333",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def main():
    fan = pd.read_csv(DATA_FAN)
    
    # Process each week
    results = []
    for (s, w), g in fan.groupby(["season", "week"]):
        if len(g) < 3:
            continue
        g = g.copy()
        n = len(g)
        
        # Shares
        judge_sum = g["judge_total"].sum()
        g["judge_share"] = g["judge_total"] / judge_sum
        g["fan_share"] = g["fan_vote_share"]
        
        # Ranks (1 = best)
        g["judge_rank"] = g["judge_total"].rank(ascending=False, method="average")
        g["fan_rank"] = g["fan_share"].rank(ascending=False, method="average")
        
        # Combined scores
        # Rank method: sum of ranks (lower = safer)
        g["rank_score"] = g["judge_rank"] + g["fan_rank"]
        # Pct method: sum of shares (higher = safer)
        g["pct_score"] = g["judge_share"] + g["fan_share"]
        
        # Safety ranking by each method (1 = safest, n = most dangerous)
        g["rank_safety"] = g["rank_score"].rank(ascending=True, method="average")
        g["pct_safety"] = g["pct_score"].rank(ascending=False, method="average")
        
        # Normalize to 0-1 (0 = most dangerous, 1 = safest)
        g["rank_norm"] = (n - g["rank_safety"]) / (n - 1) if n > 1 else 0.5
        g["pct_norm"] = (n - g["pct_safety"]) / (n - 1) if n > 1 else 0.5
        
        for _, row in g.iterrows():
            results.append({
                "season": s,
                "week": w,
                "name": row["celebrity_name"],
                "judge_rank": row["judge_rank"],
                "fan_rank": row["fan_rank"],
                "judge_share": row["judge_share"],
                "fan_share": row["fan_share"],
                "rank_norm": row["rank_norm"],
                "pct_norm": row["pct_norm"],
                "eliminated": row["eliminated_this_week"],
                "n": n
            })
    
    df = pd.DataFrame(results)
    
    alive = df[df["eliminated"] == False]
    elim = df[df["eliminated"] == True]
    
    # ============================================
    # Figure: Direct comparison of two methods
    # ============================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=800)
    
    np.random.seed(42)
    jitter = 0.012
    
    # Left plot: Safety comparison
    ax1 = axes[0]
    ax1.scatter(
        alive["rank_norm"] + np.random.uniform(-jitter, jitter, len(alive)),
        alive["pct_norm"] + np.random.uniform(-jitter, jitter, len(alive)),
        s=12, facecolors="none", edgecolors="#2b6cb0",
        alpha=0.35, linewidths=0.5, marker="o", label="Survived"
    )
    ax1.scatter(
        elim["rank_norm"] + np.random.uniform(-jitter, jitter, len(elim)),
        elim["pct_norm"] + np.random.uniform(-jitter, jitter, len(elim)),
        s=35, c="#d62728", alpha=0.8, marker="X",
        linewidths=0.5, label="Eliminated", zorder=3
    )
    
    ax1.plot([0, 1], [0, 1], color="#888888", linewidth=1.5, linestyle="--", alpha=0.7)
    
    # Highlight disagreement zones
    ax1.fill_between([0, 0.35], [0.35, 0.35], [1.02, 1.02], alpha=0.12, color="#27ae60")
    ax1.fill_between([0.35, 1.02], [0, 0], [0.35, 0.35], alpha=0.12, color="#e67e22")
    
    ax1.annotate("Pct saves\n(Fan favorites)", xy=(0.08, 0.75), fontsize=10, color="#27ae60", fontweight="bold")
    ax1.annotate("Rank saves", xy=(0.7, 0.12), fontsize=10, color="#e67e22", fontweight="bold")
    
    ax1.set_xlabel("Rank Method: Safety Score (0=Danger, 1=Safe)", fontsize=11)
    ax1.set_ylabel("Percentage Method: Safety Score (0=Danger, 1=Safe)", fontsize=11)
    ax1.set_title("Method Comparison: Who Do They Protect?", fontsize=13, fontweight="bold")
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="lower right", fontsize=9)
    
    # Right plot: Focus on "fan favorites with poor judge scores"
    ax2 = axes[1]
    
    # Define fan favorites: fan_rank better than judge_rank
    df["is_fan_favorite"] = df["fan_rank"] < df["judge_rank"]
    df["rank_diff"] = df["judge_rank"] - df["fan_rank"]  # positive = fan favorite
    
    # Calculate which method ranks them safer
    df["pct_advantage"] = df["pct_norm"] - df["rank_norm"]  # positive = pct is more lenient
    
    fan_fav = df[df["is_fan_favorite"]].copy()
    not_fav = df[~df["is_fan_favorite"]].copy()
    
    ax2.scatter(
        not_fav["rank_diff"] + np.random.uniform(-0.2, 0.2, len(not_fav)),
        not_fav["pct_advantage"] + np.random.uniform(-0.01, 0.01, len(not_fav)),
        s=10, facecolors="none", edgecolors="#95a5a6",
        alpha=0.25, linewidths=0.4, marker="o", label="Judge Favorites"
    )
    ax2.scatter(
        fan_fav["rank_diff"] + np.random.uniform(-0.2, 0.2, len(fan_fav)),
        fan_fav["pct_advantage"] + np.random.uniform(-0.01, 0.01, len(fan_fav)),
        s=12, facecolors="none", edgecolors="#3498db",
        alpha=0.4, linewidths=0.5, marker="o", label="Fan Favorites"
    )
    
    # Highlight eliminated contestants
    elim_fav = fan_fav[fan_fav["eliminated"] == True]
    ax2.scatter(
        elim_fav["rank_diff"] + np.random.uniform(-0.2, 0.2, len(elim_fav)),
        elim_fav["pct_advantage"] + np.random.uniform(-0.01, 0.01, len(elim_fav)),
        s=40, c="#d62728", alpha=0.85, marker="X",
        linewidths=0.5, label="Fan Fav. Eliminated", zorder=4
    )
    
    ax2.axhline(y=0, color="#888888", linewidth=1, linestyle="--", alpha=0.7)
    ax2.axvline(x=0, color="#888888", linewidth=1, linestyle="--", alpha=0.7)
    
    ax2.fill_between([-8, 0], [0, 0], [0.6, 0.6], alpha=0.08, color="#e67e22")
    ax2.fill_between([0, 8], [0, 0], [0.6, 0.6], alpha=0.1, color="#27ae60")
    
    ax2.annotate("Pct Method\nMore Lenient", xy=(4, 0.35), fontsize=10, color="#27ae60", ha="center")
    ax2.annotate("Rank Method\nMore Lenient", xy=(4, -0.25), fontsize=10, color="#e67e22", ha="center")
    
    ax2.set_xlabel("Fan Advantage = Judge Rank - Fan Rank\n(Positive = Fan Favorite)", fontsize=11)
    ax2.set_ylabel("Pct Advantage = Pct Safety - Rank Safety\n(Positive = Pct More Lenient)", fontsize=11)
    ax2.set_title("Does Percentage Method Favor Fan Favorites?", fontsize=13, fontweight="bold")
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "method_comparison_analysis.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "method_comparison_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    
    # Statistics
    print("=== Analysis ===")
    fan_fav_stats = fan_fav.copy()
    print(f"Fan Favorites (fan_rank < judge_rank): {len(fan_fav_stats)}")
    print(f"  - Pct more lenient: {(fan_fav_stats['pct_advantage'] > 0).sum()} ({(fan_fav_stats['pct_advantage'] > 0).mean()*100:.1f}%)")
    print(f"  - Mean Pct Advantage: {fan_fav_stats['pct_advantage'].mean():.3f}")
    
    print(f"\nSaved: {OUT_DIR / 'method_comparison_analysis.png'}")


if __name__ == "__main__":
    main()
