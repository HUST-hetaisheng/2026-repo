import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "rank": "#1f77b4",
    "percent": "#2ca02c",
    "bottom2": "#ff7f0e",
}

REGIME_LABELS = {
    "rank": "Rank",
    "percent": "Percent",
    "bottom2": "Bottom-2",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
    })


def add_regime_shading(ax):
    ax.axvspan(0.5, 2.5, color="#e6eef6", alpha=0.7, zorder=0)
    ax.axvspan(2.5, 27.5, color="#eef7ef", alpha=0.7, zorder=0)
    ax.axvspan(27.5, 34.5, color="#fff1e6", alpha=0.75, zorder=0)


def plot_ci_width_by_season():
    df = pd.read_csv(DATA_DIR / "season_uncertainty_summary.csv")
    df = df.sort_values("season")
    df["season"] = df["season"].astype(int)
    df["ci_width"] = pd.to_numeric(df["ci_width"], errors="coerce")
    df["cv"] = pd.to_numeric(df["cv"], errors="coerce")

    x = df["season"].values
    y = df["ci_width"].values
    std = df["cv"].values * y
    lower = np.clip(y - std, 0, None)
    upper = y + std

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    add_regime_shading(ax)

    ax.fill_between(x, lower, upper, color="#3d3d3d", alpha=0.15, zorder=1)
    ax.plot(x, y, color="#222222", linewidth=2.4, zorder=3)
    ax.scatter(
        x,
        y,
        s=28,
        color=df["regime"].map(COLORS),
        edgecolor="white",
        linewidth=0.4,
        zorder=4,
    )

    ax.axhline(0.01, linestyle="--", color="#666666", linewidth=1)
    ax.set_title("Spatio-Temporal Evolution of CI Width")
    ax.set_xlabel("Season")
    ax.set_ylabel("Uncertainty Magnitude (Avg CI Width)")
    ax.set_xlim(0.5, 34.5)
    ax.set_xticks(list(range(1, 35, 2)))

    y_max = max(upper.max(), 0.012)
    ax.set_ylim(0, y_max * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.18)

    ax.text(
        0.02,
        0.94,
        "Rank Era (High Uncertainty)",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#4b6a8a",
    )
    ax.text(
        0.38,
        0.94,
        "Percent Era (High Precision)",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#2f6b43",
    )
    ax.text(
        0.74,
        0.94,
        "Judge Save Era (Structural Ambiguity)",
        transform=ax.transAxes,
        fontsize=9.6,
        color="#8a5a2a",
    )

    legend_handles = [
        Line2D([0], [0], color="#222222", lw=2.2, label="Avg CI Width"),
        Patch(facecolor=COLORS["rank"], label=REGIME_LABELS["rank"]),
        Patch(facecolor=COLORS["percent"], label=REGIME_LABELS["percent"]),
        Patch(facecolor=COLORS["bottom2"], label=REGIME_LABELS["bottom2"]),
        Line2D([0], [0], color="#666666", lw=1, linestyle="--", label="1% Margin Threshold"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ci_width_by_season.png", dpi=1000)


def plot_ci_width_by_week():
    df = pd.read_csv(DATA_DIR / "week_uncertainty_summary.csv")
    df = df.sort_values("week")
    df["week"] = df["week"].astype(int)
    df["ci_width"] = pd.to_numeric(df["ci_width"], errors="coerce")
    df["cv"] = pd.to_numeric(df["cv"], errors="coerce")

    x = df["week"].values
    y_width = df["ci_width"].values
    y_cv = df["cv"].values

    cmap = mpl.colormaps.get_cmap("coolwarm")
    colors = cmap(np.linspace(0.1, 0.9, len(x)))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, y_cv, color=colors, edgecolor="none", alpha=0.85, width=0.7, zorder=2)
    ax.set_xlabel("Week of Competition")
    ax.set_ylabel("Relative Volatility (CV)")
    ax.set_xlim(0.5, 11.5)
    ax.set_xticks(range(1, 12))
    ax.set_ylim(0, max(y_cv) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.18)

    ax2 = ax.twinx()
    std = y_cv * y_width
    lower = np.clip(y_width - std, 0, None)
    upper = y_width + std
    ax2.fill_between(x, lower, upper, color="#222222", alpha=0.15, zorder=1)
    ax2.plot(x, y_width, color="#222222", linewidth=2.4, zorder=3)
    ax2.scatter(x, y_width, s=26, color=colors, edgecolor="white", linewidth=0.3, zorder=4)
    ax2.set_ylim(0, max(y_width) * 1.35)
    ax2.set_ylabel("Uncertainty Magnitude (CI Width)")

    ax.set_title("The Clarity Horizon: Temporal Evolution of Inference Confidence")

    label = "Uncertainty Convergence" if y_cv[-1] < y_cv[0] else "Persistent Ambiguity"
    ax.annotate(
        label,
        xy=(x[-1], y_cv[-1]),
        xycoords="data",
        xytext=(0.18, 0.86),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#444444", linewidth=1),
        fontsize=10,
        color="#444444",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ci_width_by_week.png", dpi=1000)


def main():
    apply_style()
    plot_ci_width_by_season()
    plot_ci_width_by_week()


if __name__ == "__main__":
    main()
