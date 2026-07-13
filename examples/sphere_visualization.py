from fibonacci_lattice_tiling_toolkit.utilities.data_utils import compute_CMP_tile_areas, get_CMP_tile_boundaries, get_FB_tile_boundaries, get_ERP_tile_boundaries
from fibonacci_lattice_tiling_toolkit.data_types import Vector
from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_tiling_visualization_image, save_tiling_visualization_video
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.image as mpimg

erp_image_path = "../input/world_ERP.jpg"

def save_erp_2d_grid(num_tiles_h, num_tiles_v, output_path, erp_image_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Set the plot limits to match standard Lat/Lon coordinates
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # 2. Add the ERP background image
    if erp_image_path:
        img = mpimg.imread(erp_image_path)
        # 'extent' maps the image edges to the specific axis coordinates
        ax.imshow(img, extent=[-180, 180, -90, 90], aspect='auto')

    width = 360 / num_tiles_h
    height = 180 / num_tiles_v
    colormap = cm.get_cmap('Greys') 

    # 3. Draw the grid tiles
    for i in range(num_tiles_h):
        for j in range(num_tiles_v):
            x = -180 + (i * width)
            y = -90 + (j * height)
            
            intensity = 0 + ((num_tiles_v - j) / (num_tiles_v - 1)) * 0.2
            tile_color = colormap(intensity)

            rect = patches.Rectangle(
                (x, y), width, height, 
                linewidth=1.5,      # Reduced linewidth for better visibility over images
                edgecolor='black', 
                facecolor='none',
                alpha=1.0           # Added transparency to see the image underneath
            )
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save with no white borders
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


ERP_tile_list = [(22, 11)] #(10, 5), (16, 8), (22, 11), (30, 15), (40, 20), (60, 30), (80, 40)]

azimuth_elevation_list = [(0, 0)]

camera_positions = [(5, 0, 1)]

tile_counts = []

output_dir = Path("../output")
# Ensure the directory exists
output_dir.mkdir(parents=True, exist_ok=True)

for num_tiles in ERP_tile_list:
    # FB Tile Count
    FB_tile_count = num_tiles[0] * num_tiles[1]

    if (FB_tile_count % 2 == 0):
        FB_tile_count -= 1

    # ERP Tile Count
    ERP_num_tiles_horizontal = num_tiles[0]
    ERP_num_tiles_vertical = num_tiles[1]
    ERP_tile_count = ERP_num_tiles_horizontal * ERP_num_tiles_vertical

    h, v = num_tiles
    erp_filename = output_dir / f"ERP_layout_{h}x{v}.png"
    save_erp_2d_grid(h, v, erp_filename, erp_image_path=erp_image_path)


    # CMP Tile Count
    tiles_per_CMP_face = ERP_tile_count / 6.0
    CMP_num_tiles_horizontal = int(np.ceil(np.sqrt(tiles_per_CMP_face)))
    CMP_num_tiles_vertical = CMP_num_tiles_horizontal
    CMP_tile_count = CMP_num_tiles_horizontal * CMP_num_tiles_vertical * 6

    # Generate tiles for Fibonacci lattice tiling, ERP tiling, and CMP tiling
    # FB_tile_boundaries = get_FB_tile_boundaries(FB_tile_count)
    ERP_tile_boundaries = get_ERP_tile_boundaries(num_tiles_horizontal=ERP_num_tiles_horizontal, num_tiles_vertical=ERP_num_tiles_vertical)
    # CMP_tile_boundaries = get_CMP_tile_boundaries(num_tiles_horizontal=CMP_num_tiles_horizontal, num_tiles_vertical=CMP_num_tiles_vertical)

    for azimuth_elevation in azimuth_elevation_list:
        camera_azimuth = azimuth_elevation[0]
        camera_elevation = azimuth_elevation[1]

        for camera_position in camera_positions:

            output_prefix = f"azimuth_{camera_azimuth}-elevation_{camera_elevation}_FB_({FB_tile_count})_tiling-"

            save_tiling_visualization_image(
                tile_boundaries=FB_tile_boundaries,
                output_dir=output_dir,
                output_prefix=output_prefix,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                camera_position=camera_position)

            output_prefix = f"azimuth_{camera_azimuth}-elevation_{camera_elevation}_ERP_({ERP_tile_count})_tiling-"

            save_tiling_visualization_image(
                tile_boundaries=ERP_tile_boundaries,
                output_dir=output_dir,
                erp_image_path=erp_image_path,
                output_prefix=output_prefix,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                camera_position=camera_position)

            output_prefix = f"azimuth_{camera_azimuth}-elevation_{camera_elevation}_CMP_({CMP_tile_count})_tiling-"

            save_tiling_visualization_image(
                tile_boundaries=CMP_tile_boundaries,
                output_dir=output_dir,
                output_prefix=output_prefix,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                camera_position=camera_position)