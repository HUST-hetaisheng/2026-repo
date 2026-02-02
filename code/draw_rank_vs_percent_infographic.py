import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def apply_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.titlesize": 16,
    })


def draw_panel(ax, theme):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=theme["bg"],
        edgecolor=theme["edge"],
        linewidth=1.0,
    )
    ax.add_patch(panel)

    # Header bar
    ax.add_patch(Rectangle((0.02, 0.93), 0.96, 0.06, facecolor=theme["header"], edgecolor="none"))
    ax.text(0.5, 0.96, theme["title"], ha="center", va="center", fontsize=11.5, color="white", weight="bold")


def draw_wave(ax, x0, y0, width, height, color, peaks):
    xs = np.linspace(0, 1, 200)
    ys = np.zeros_like(xs)
    for pos, amp, spread in peaks:
        ys += amp * np.exp(-0.5 * ((xs - pos) / spread) ** 2)
    ys = ys / ys.max()
    xs = x0 + width * xs
    ys = y0 + height * ys
    ax.fill_between(xs, y0, ys, color=color, alpha=0.85, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=1.6)


def draw_staircase(ax, x0, y0, width, height, steps):
    step_w = width / steps
    for i in range(steps):
        ax.add_patch(Rectangle((x0 + i * step_w, y0), step_w, height * (steps - i) / steps,
                               facecolor="#cfd6dd", edgecolor="#7f8a94", linewidth=1.0))
        ax.text(x0 + i * step_w + step_w / 2, y0 + 0.02, f"{i+1}th" if i > 0 else "1st",
                ha="center", va="bottom", fontsize=8, color="#4a4a4a")


def draw_rank_panel(ax):
    draw_panel(ax, {
        "bg": "#edf2f8",
        "edge": "#b7c3d1",
        "header": "#7c93ad",
        "title": "THE RANK METHOD",
    })

    ax.text(0.06, 0.84, "Raw Fan Votes\n(Input)", ha="left", va="center", fontsize=9, color="#3e4a57")
    draw_wave(ax, 0.2, 0.78, 0.55, 0.12, "#9a5cc7",
              [(0.15, 0.4, 0.07), (0.42, 1.0, 0.06), (0.7, 0.5, 0.05), (0.85, 0.35, 0.05)])

    # Rank transformation device
    ax.add_patch(FancyBboxPatch((0.72, 0.78), 0.18, 0.11, boxstyle="round,pad=0.01",
                                facecolor="#dfe7ef", edgecolor="#8a98a7", linewidth=1))
    ax.text(0.81, 0.835, "Rank\nTransformation", ha="center", va="center", fontsize=8, color="#3e4a57")
    ax.add_patch(FancyArrowPatch((0.62, 0.84), (0.72, 0.84), arrowstyle="->", linewidth=1, color="#54606c"))

    ax.text(0.06, 0.58, "Staircase of Ranks\n(Output)", ha="left", va="center", fontsize=9, color="#3e4a57")
    draw_staircase(ax, 0.18, 0.46, 0.62, 0.18, steps=5)
    ax.text(0.18, 0.42, "Compression: Large leads become uniform steps.", fontsize=8, color="#5b6773")

    ax.text(0.06, 0.22, "Impact on Outcome", ha="left", va="center", fontsize=9, color="#3e4a57", weight="bold")
    ax.plot([0.18, 0.62], [0.18, 0.18], color="#6c7a88", linewidth=2)
    ax.add_patch(Circle((0.2, 0.18), 0.025, facecolor="#6c7a88", edgecolor="none"))
    ax.text(0.24, 0.13, "JUDGE'S SCORES", fontsize=8, color="#4a4a4a")
    ax.add_patch(FancyArrowPatch((0.35, 0.18), (0.48, 0.18), arrowstyle="->", linewidth=1, color="#6c7a88"))
    ax.text(0.46, 0.13, "ELIMINATED", fontsize=8, color="#a4493d", weight="bold")
    ax.add_patch(FancyArrowPatch((0.55, 0.18), (0.66, 0.18), arrowstyle="->", linewidth=1, color="#6c7a88"))
    ax.text(0.66, 0.13, "FANS", fontsize=8, color="#4a4a4a")

    ax.text(0.5, 0.04, "Equal Steps, Lost Magnitude", ha="center", va="center", fontsize=9.5, color="#2f4152")


def draw_percentage_panel(ax):
    draw_panel(ax, {
        "bg": "#fbf4e8",
        "edge": "#d7c3a3",
        "header": "#d2a35b",
        "title": "THE PERCENTAGE METHOD",
    })

    ax.text(0.06, 0.84, "Percentage Preservation\n(Proportional)", ha="left", va="center", fontsize=9, color="#6c4b1f")
    draw_wave(ax, 0.2, 0.78, 0.55, 0.12, "#f08aa7",
              [(0.15, 0.35, 0.07), (0.42, 1.0, 0.06), (0.7, 0.45, 0.05), (0.85, 0.3, 0.05)])

    ax.text(0.06, 0.58, "Proportional Landscape\n(Output)", ha="left", va="center", fontsize=9, color="#6c4b1f")
    draw_wave(ax, 0.2, 0.48, 0.6, 0.16, "#8c3ad9",
              [(0.2, 0.6, 0.08), (0.5, 1.0, 0.07), (0.8, 0.5, 0.07)])
    # Magnifier
    ax.add_patch(Circle((0.78, 0.55), 0.06, edgecolor="#6c4b1f", facecolor="none", linewidth=1.2))
    ax.add_patch(Rectangle((0.82, 0.50), 0.05, 0.015, angle=25, facecolor="#6c4b1f", edgecolor="none"))

    ax.text(0.06, 0.22, "Impact on Outcome", ha="left", va="center", fontsize=9, color="#6c4b1f", weight="bold")
    ax.plot([0.18, 0.62], [0.18, 0.18], color="#9c7a42", linewidth=2)
    ax.add_patch(Circle((0.2, 0.18), 0.025, facecolor="#9c7a42", edgecolor="none"))
    ax.text(0.24, 0.13, "JUDGE'S SCORES", fontsize=8, color="#6c4b1f")
    ax.add_patch(FancyArrowPatch((0.35, 0.18), (0.48, 0.18), arrowstyle="->", linewidth=1, color="#9c7a42"))
    ax.text(0.47, 0.13, "SAVED", fontsize=8, color="#2c8c4d", weight="bold")
    ax.add_patch(FancyArrowPatch((0.55, 0.18), (0.66, 0.18), arrowstyle="->", linewidth=1, color="#9c7a42"))
    ax.text(0.66, 0.13, "FANS", fontsize=8, color="#6c4b1f")

    ax.text(0.5, 0.04, "True Proportions Preserved", ha="center", va="center", fontsize=9.5, color="#7a4e1d")


def build_infographic():
    apply_style()
    fig = plt.figure(figsize=(14, 7))

    left = fig.add_axes([0.05, 0.12, 0.43, 0.76])
    right = fig.add_axes([0.52, 0.12, 0.43, 0.76])

    draw_rank_panel(left)
    draw_percentage_panel(right)

    fig.suptitle("Judges VS Fans: The Battle for Power in DWTS Voting", y=0.96, fontsize=18, weight="bold")

    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.axis("off")
    overlay.text(0.5, 0.52, "From Ordinal to Cardinal", ha="center", va="center", fontsize=11, color="#444444")
    overlay.text(0.5, 0.485, "How the same votes can tip the scale differently", ha="center", va="center",
                 fontsize=9.5, color="#555555")
    overlay.add_patch(FancyArrowPatch((0.45, 0.52), (0.55, 0.52), arrowstyle="->",
                                      mutation_scale=12, linewidth=1.2, color="#444444"))

    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.08)
    fig.savefig(FIG_DIR / "judges_vs_fans_rank_vs_percent.png", dpi=1000)


if __name__ == "__main__":
    build_infographic()
