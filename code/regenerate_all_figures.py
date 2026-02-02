#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量重新生成所有图片 (1000 dpi)
"""

import subprocess
import sys
import os

# 切换到 code 目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 需要运行的脚本列表
scripts = [
    # code 目录下的脚本
    "draw_cv_density_by_regime.py",
    "plot_controversial_rank_scatter.py",
    "plot_rank_diff_vs_placement.py",
    "plot_rank_pct_vs_placement.py",
    "plot_counterfactual_survival.py",
    "plot_tolerance_frontier.py",
    "plot_rank_vs_pct_bias.py",
    "plot_fate_reversal.py",
    "plot_safety_margin_beautiful.py",
    "task3_visualization_v4.py",
    "task3_visualization_final.py",
    "task3_visualization.py",
    "task3_industry_analysis.py",
    "judge_independence_visualization.py",
    "draw_rules_chart.py",
    "draw_rank_vs_percent_infographic.py",
    "draw_rank_vs_percent_concept.py",
    "draw_consistency_uncertainty.py",
    "draw_ci_uncertainty_advanced.py",
]

# 根目录和 polish 目录的脚本
root_scripts = [
    ("../plot_rank_pct.py", ".."),
    ("../polish/run_survival_simulation.py", "../polish"),
    ("../polish/task3_mixed_effects_visualization.py", "../polish"),
]

print("=" * 70)
print("批量重新生成所有图片 (1000 dpi)")
print("=" * 70)

failed = []

# 运行 code 目录下的脚本
for script in scripts:
    print(f"\n[运行] {script}")
    try:
        result = subprocess.run([sys.executable, script], 
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ⚠ 警告: {script} 返回非零状态")
            if result.stderr:
                print(f"  错误: {result.stderr[:200]}")
            failed.append(script)
        else:
            print(f"  ✓ 完成")
    except subprocess.TimeoutExpired:
        print(f"  ⚠ 超时: {script}")
        failed.append(script)
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        failed.append(script)

# 运行其他目录的脚本
for script, cwd in root_scripts:
    print(f"\n[运行] {script}")
    try:
        result = subprocess.run([sys.executable, os.path.basename(script)], 
                                cwd=os.path.join(os.path.dirname(__file__), cwd),
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ⚠ 警告: {script} 返回非零状态")
            if result.stderr:
                print(f"  错误: {result.stderr[:200]}")
            failed.append(script)
        else:
            print(f"  ✓ 完成")
    except subprocess.TimeoutExpired:
        print(f"  ⚠ 超时: {script}")
        failed.append(script)
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        failed.append(script)

print("\n" + "=" * 70)
print(f"完成! 成功: {len(scripts) + len(root_scripts) - len(failed)}, 失败: {len(failed)}")
if failed:
    print("失败的脚本:")
    for f in failed:
        print(f"  - {f}")
print("=" * 70)
