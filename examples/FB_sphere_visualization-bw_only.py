from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_fb_tiling_visualization_glb
from pathlib import Path
from FB_projection_visualization import get_connections

tile_counts = [65]

# Visualization settings
wrap = False
lift_bottom = False
columns = True
rows = True

if __name__ == "__main__":
    output_dir = Path("../output")
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for num_tiles in tile_counts:

        if (num_tiles % 2 == 0):
            num_tiles -= 1

        input_dir = Path("../input")
        layout_file_name = f"SFLP_{num_tiles}_rc_spirals_layout.csv"
        layout_csv_path = input_dir / layout_file_name

        

        connections = get_connections(
            layout_csv_path=layout_csv_path,
            columns=columns,
            rows=rows,
            wrap=wrap,
            lift_bottom=lift_bottom,
        )

        save_fb_tiling_visualization_glb(
            tile_count=num_tiles,
            output_dir=output_dir,
            connections=connections,
            include_indices=True,
            bw_only=True,
        )