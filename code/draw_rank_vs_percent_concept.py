import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def apply_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 16,
    })


def draw_scale(ax, theme):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 10, 8.5, facecolor=theme["bg"], edgecolor="none", zorder=0))

    ax.add_patch(Polygon([(5, 0.6), (3.6, 0), (6.4, 0)], closed=True, facecolor="#555555", zorder=2))
    ax.add_patch(Rectangle((4.85, 0.6), 0.3, 4.9, facecolor="#666666", zorder=2))
    ax.plot([2.0, 8.0], [6.2, 6.2], color="#444444", linewidth=4, zorder=2)

    ax.plot([2.5, 2.5], [6.2, 5.15], color="#444444", linewidth=2, zorder=2)
    ax.plot([7.5, 7.5], [6.2, 5.15], color="#444444", linewidth=2, zorder=2)

    ax.add_patch(Ellipse((2.5, 4.95), 2.5, 0.5, facecolor="#e5e5e5", edgecolor="#777777", zorder=2))
    ax.add_patch(Ellipse((7.5, 4.95), 2.5, 0.5, facecolor="#e5e5e5", edgecolor="#777777", zorder=2))

    ax.text(2.5, 6.75, "Judges", ha="center", va="bottom", fontsize=11, color="#333333")
    ax.text(7.5, 6.75, "Audience", ha="center", va="bottom", fontsize=11, color="#333333")


def draw_weight(ax, center, size, label, facecolor, edgecolor="#2f4152"):
    x, y = center
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=10, color=edgecolor, zorder=4)


def draw_beaker(ax, center, width, height, fill_ratio, outline="#b56f2a", liquid="#f4b463"):
    x, y = center
    ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height, facecolor="none", edgecolor=outline, linewidth=1.3, zorder=3))
    liquid_height = height * fill_ratio
    ax.add_patch(
        Rectangle(
            (x - width / 2 + 0.05, y - height / 2 + 0.05),
            width - 0.1,
            max(liquid_height - 0.1, 0),
            facecolor=liquid,
            edgecolor="none",
            alpha=0.85,
            zorder=2,
        )
    )


def build_democracy_scale():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))

    rank_theme = {"bg": "#eef3f8"}
    percent_theme = {"bg": "#faf5ee"}

    # Left panel: Rank
    draw_scale(axes[0], rank_theme)
    axes[0].set_title("Rank Method")
    draw_weight(axes[0], (2.5, 5.35), 0.95, "1", facecolor="#cfe1f2")
    draw_weight(axes[0], (7.5, 5.35), 0.95, "2", facecolor="#cfe1f2")
    draw_weight(axes[0], (2.0, 4.55), 0.75, "3", facecolor="#d9e7f5")
    draw_weight(axes[0], (8.0, 4.55), 0.75, "4", facecolor="#d9e7f5")
    axes[0].text(5, 1.0, "Equal Steps, Lost Magnitude", ha="center", va="bottom", fontsize=10, color="#2f4152")

    # Right panel: Percentage
    draw_scale(axes[1], percent_theme)
    axes[1].set_title("Percentage Method")
    draw_beaker(axes[1], (2.5, 5.1), width=1.3, height=1.6, fill_ratio=0.3)
    draw_beaker(axes[1], (7.5, 5.1), width=1.3, height=1.6, fill_ratio=0.6)
    axes[1].text(5, 1.0, "True Proportions Preserved", ha="center", va="bottom", fontsize=10, color="#7a4e1d")

    fig.suptitle("The Democracy Scale: Rank vs Percentage", y=0.96, fontsize=18)

    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    overlay.axis("off")
    overlay.annotate(
        "From Ordinal to Cardinal",
        xy=(0.5, 0.53),
        xytext=(0.5, 0.53),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )
    overlay.add_patch(
        FancyArrowPatch(
            (0.43, 0.53),
            (0.57, 0.53),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            color="#444444",
            transform=overlay.transAxes,
        )
    )
    overlay.annotate(
        "How the same votes can tip the scale differently",
        xy=(0.5, 0.47),
        xytext=(0.5, 0.47),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#555555",
    )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12, wspace=0.12)
    fig.savefig(FIG_DIR / "democracy_scale_rank_vs_percent.png", dpi=600)


def gradient_fill(ax, x, y, color_left, color_right, alpha=0.6):
    x = np.asarray(x)
    y = np.asarray(y)
    ymin = 0
    verts = [(x[0], ymin)] + list(zip(x, y)) + [(x[-1], ymin)]
    path = MplPath(verts)

    grad = np.linspace(0, 1, 256)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("grad", [color_left, color_right])
    im = ax.imshow(
        grad.reshape(1, -1),
        extent=[x.min(), x.max(), ymin, max(y) * 1.15],
        aspect="auto",
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        zorder=1,
    )
    im.set_clip_path(path, transform=ax.transData)


def build_compression_echo():
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True)

    contestants = ["A", "B", "C", "D", "E"]
    x = np.arange(len(contestants))
    votes = np.array([0.35, 0.25, 0.20, 0.12, 0.08])

    # Panel 1: Raw
    ax = axes[0]
    gradient_fill(ax, x, votes, "#b36bff", "#ff6fb7", alpha=0.75)
    ax.plot(x, votes, color="#7c2ae8", linewidth=2.2, zorder=2)
    ax.scatter(x, votes, color="#ff6fb7", s=30, zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Vote Share")
    ax.set_title("Raw Fan Votes")
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    # Panel 2: Rank
    ax = axes[1]
    ranks = np.array([5, 4, 3, 2, 1])
    ax.step(x, ranks, where="mid", color="#666666", linewidth=2.2, zorder=2)
    ax.fill_between(x, ranks, step="mid", color="#cfcfcf", alpha=0.8, zorder=1)
    ax.plot(x, votes * 12, color="#b36bff", linewidth=1.5, alpha=0.25, zorder=0)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("Rank Points")
    ax.set_title("After Rank Transformation")
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    # Panel 3: Percent
    ax = axes[2]
    smooth_x = np.linspace(x.min(), x.max(), 200)
    smooth_y = np.interp(smooth_x, x, votes)
    gradient_fill(ax, smooth_x, smooth_y, "#b36bff", "#ff6fb7", alpha=0.5)
    ax.plot(smooth_x, smooth_y, color="#7c2ae8", linewidth=2.2, zorder=2)
    ax.scatter(x, votes, color="#ff6fb7", s=28, zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("Contribution Score")
    ax.set_title("After Percentage Preservation")
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(contestants)
    axes[2].set_xlabel("Contestants")

    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    overlay.axis("off")
    overlay.annotate(
        "Compression",
        xy=(0.5, 0.6),
        xytext=(0.5, 0.68),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1),
    )
    overlay.annotate(
        "Preservation",
        xy=(0.5, 0.33),
        xytext=(0.5, 0.41),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1),
    )
    fig.suptitle("The Loudest Voice: How Rank Silences Superstars", y=0.97, fontsize=18)

    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.08, hspace=0.5)
    fig.savefig(FIG_DIR / "compression_echo_rank_vs_percent.png", dpi=600)


def build_terrain_comparison():
    apply_style()
    fig = plt.figure(figsize=(11.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")

    contestants = ["A", "B", "C", "D", "E"]
    votes = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
    x_centers = np.arange(len(contestants))
    y_centers = np.array([1.0] * len(contestants))

    x = np.linspace(-0.5, len(contestants) - 0.5, 180)
    y = np.linspace(0.0, 6.0, 160)
    X, Y = np.meshgrid(x, y)

    sigma_x = 0.35
    sigma_y = 0.55

    Z_back = np.zeros_like(X)
    Z_front = np.zeros_like(X)

    # Rank levels (equal steps)
    ranks = np.array([5, 4, 3, 2, 1], dtype=float)
    rank_levels = ranks / ranks.max()

    for i, (cx, cy) in enumerate(zip(x_centers, y_centers)):
        height = votes[i] / votes.max()
        gaussian = np.exp(-((X - cx) ** 2 / (2 * sigma_x**2) + (Y - (cy + 3.2)) ** 2 / (2 * sigma_y**2)))
        Z_back += height * gaussian

        rank_height = rank_levels[i]
        gaussian_front = np.exp(-((X - cx) ** 2 / (2 * sigma_x**2) + (Y - cy) ** 2 / (2 * sigma_y**2)))
        Z_front += rank_height * gaussian_front

    # Back: Percentage world
    ax.plot_surface(
        X,
        Y,
        Z_back,
        cmap=mpl.cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        zorder=1,
    )

    # Front: Rank world (stepped peaks)
    ax.plot_surface(
        X,
        Y,
        Z_front,
        cmap=mpl.cm.cividis,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        zorder=2,
    )

    # Labels for back range
    for i, cx in enumerate(x_centers):
        label = f"{int(votes[i]*100)}%"
        ax.text(cx, 3.3, Z_back.max() * (votes[i] / votes.max()) + 0.03, label, color="#1b2c2f", fontsize=9)

    ax.text(x_centers[0], 3.8, Z_back.max() * 1.05, "Fan Favorite (35%)", color="#0f2f2f", fontsize=10)

    # Region titles
    ax.text(-0.8, 4.9, Z_back.max() * 0.9, "Percentage World", color="#1f4b3a", fontsize=11)
    ax.text(-0.8, 0.6, Z_back.max() * 0.7, "Rank World", color="#4d4d4d", fontsize=11)

    ax.set_title("Peaks vs Steps: Rank and Percentage Worlds", pad=10)
    ax.view_init(elev=28, azim=-60)

    ax.set_xlim(-0.5, len(contestants) - 0.2)
    ax.set_ylim(-0.2, 6.2)
    ax.set_zlim(0, Z_back.max() * 1.2)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(contestants)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06)
    fig.savefig(FIG_DIR / "terrain_rank_vs_percent.png", dpi=600)


if __name__ == "__main__":
    build_democracy_scale()
    build_compression_echo()
    build_terrain_comparison()
