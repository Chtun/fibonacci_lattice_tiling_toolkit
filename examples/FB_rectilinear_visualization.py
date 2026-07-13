import matplotlib.pyplot as plt
import pandas
from pathlib import Path


FB_count = 65
layout_file_name = f"SFLP_{FB_count}_layout.csv"
output_file_name = f"FB_Projection-({FB_count})_points-rectilinear_arrangement.png"
graph_title = "SFLP in Final, Proposed Arrangement"

def plot_rectilinear_grid(
    layout_csv_path: Path,
    title: str,
    save_path: Path,
    figsize: tuple[float, float] = (12, 12),
    circle_size: int = 2800,
    line_width: int = 6,
    font_size: int = 32
):
    """
    Illustrates the layout as a perfect grid based on CSV structure
    rather than spatial coordinates.
    """
    # Load the layout
    layout_pd = pandas.read_csv(layout_csv_path, header=None)
    rows_count, cols_count = layout_pd.shape

    plt.figure(figsize=figsize)

    # 1. Draw Grid Lines (Rectilinear connections)
    # Draw horizontal rows
    for r in range(rows_count):
        plt.plot([0, cols_count - 1], [r, r], color='green', alpha=0.7, linewidth=line_width, zorder=1)
    
    # Draw vertical columns
    for c in range(cols_count):
        plt.plot([c, c], [0, rows_count - 1], color='red', alpha=0.7, linewidth=line_width, zorder=1)

    # 2. Plot the indices as nodes in the grid
    for r in range(rows_count):
        for c in range(cols_count):
            index_val = layout_pd.iloc[r, c]
            
            # Distinguish between real points and padded/null points
            node_color = 'blue' if index_val >= 0 else 'black'
            label_val = str(int(abs(index_val)))

            # Note: We plot 'c' as X and 'rows_count - r' as Y 
            # so the top of the CSV is the top of the plot
            x_pos = c
            y_pos = rows_count - 1 - r

            plt.scatter(x_pos, y_pos, color=node_color, s=circle_size, 
                        edgecolors='black', zorder=2)
            
            plt.text(x_pos, y_pos, label_val, color='white', 
                     fontsize=font_size, ha='center', va='center', 
                     fontweight='bold', zorder=3)

    # Formatting
    plt.title(title, fontsize=20)
    plt.xlabel('Grid Column Index', fontsize=14)
    plt.ylabel('Grid Row Index', fontsize=14)
    
    # Set limits to show the full grid comfortably
    plt.xlim(-0.5, cols_count - 0.5)
    plt.ylim(-0.5, rows_count - 0.5)
    
    plt.grid(False) # Turn off standard matplotlib grid
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rectilinear grid visualization saved to: {save_path}")





if __name__ == "__main__":
    input_dir = Path("../input")
    output_dir = Path("../output")
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_csv_path = input_dir / layout_file_name

    output_path = output_dir / output_file_name

    plot_rectilinear_grid(
        layout_csv_path=layout_csv_path,
        title=graph_title,
        save_path=output_path,
    )

