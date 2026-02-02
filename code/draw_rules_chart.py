import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os

# Ensure figures directory exists
output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set font to Times New Roman and increase base sizes for readability
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Data definition
total_seasons = 34
seasons = range(1, total_seasons + 1)

# Review prompt details:
# Scoring: 
#   "combine by rank": 1-2, 28-34
#   "combine by percentage": 3-27
# Elimination:
#   "eliminate the last": 1-27
#   "eliminate from the bottom two": 28-34

# Define tracks (Labels)
tracks = [
    "combine by rank",
    "combine by percentage", 
    "eliminate the last", 
    "eliminate from the bottom two"
]

# Define active ranges for each track (inclusive)
active_ranges = {
    "combine by rank": [(1, 2), (28, 34)],
    "combine by percentage": [(3, 27)],
    "eliminate the last": [(1, 27)],
    "eliminate from the bottom two": [(28, 34)]
}

# Define colors
# Scheme 1 (Scoring) - Blue tones
color_score_1 = '#1f77b4' # Rank - Darker Blue
color_score_2 = '#6baed6' # Percentage - Lighter Blue

# Scheme 2 (Elimination) - Red/Orange tones
color_elim_1 = '#d62728' # Last - Red
color_elim_2 = '#ff7f0e' # Bottom two - Orange

colors = {
    "combine by rank": color_score_1,
    "combine by percentage": color_score_2,
    "eliminate the last": color_elim_1,
    "eliminate from the bottom two": color_elim_2
}

gray_color = '#d3d3d3'

# Reduce figure height to shorten distance between tracks
fig, ax = plt.subplots(figsize=(16, 5.2))

# Plot tracks
y_positions = [3, 2, 1, 0] # Top to bottom

for i, track_name in enumerate(tracks):
    y = y_positions[i]
    color = colors[track_name]
    ranges = active_ranges[track_name]
    
    # We need to cover the whole timeline from season 1 to 34
    # It's easier to iterate through each season step or segments
    # Let's draw the full inactive line first
    ax.plot([1, total_seasons], [y, y], color=gray_color, linestyle='--', linewidth=4, zorder=1)
    
    # Draw active segments
    for start, end in ranges:
        # Drawing from specific season to end season
        # Using a slightly larger linewidth for emphasis
        ax.plot([start, end], [y, y], color=color, linestyle='-', linewidth=6, zorder=2)
        
        # Add markers at the ends of active segments for clarity
        ax.scatter([start, end], [y, y], color=color, s=50, zorder=3)

# Formatting
ax.set_yticks(y_positions)
ax.set_yticklabels(tracks, fontsize=13)
ax.set_ylim(-0.5, 3.5)
ax.set_xlim(0.5, 34.5)

# X-axis on top (Prompt: "轨道最上方标记周数")
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 
ax.set_xticks(range(1, 35, 1)) # Mark every season
ax.set_xlabel("Seasons", fontsize=14, labelpad=12)

# Remove spines except maybe top? or remove all box
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# ax.spines['top'].set_visible(False) 

# Grid for x-axis to see alignment better?
ax.grid(axis='x', linestyle=':', alpha=0.5)

plt.title("Rules of Score Combining & Couple Elimination", y=1.22, fontsize=20, weight='bold')

# Create legend handles
legend_handles = []
for track_name in tracks:
    legend_handles.append(mlines.Line2D([], [], color=colors[track_name], linewidth=4, label=track_name))
legend_handles.append(mlines.Line2D([], [], color=gray_color, linestyle='--', linewidth=3, label='Rule Not Applied'))

# Add legend below the plot
ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=12)

plt.tight_layout()

output_path = os.path.join(output_dir, 'rules_chart.png')
plt.savefig(output_path, dpi=1000)
print(f"Chart saved to {output_path}")
