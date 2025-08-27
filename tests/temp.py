from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_fb_tiling_visualization_image, save_fb_tiling_visualization_glb
from pathlib import Path


output_dir = Path("../output")
# Ensure the directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# save_fb_tiling_visualization_image(
#     61,
#     output_dir=output_dir,
#     camera_position=(0, 5, 5),
#     camera_up=(0, 1, 0),
#     include_indices=True,
# )

# save_fb_tiling_visualization_image(
#     161,
#     output_dir=output_dir,
#     include_indices=True,
# )

save_fb_tiling_visualization_glb(
    61,
    output_dir=output_dir,
    include_indices=True,
)

save_fb_tiling_visualization_glb(
    161,
    output_dir=output_dir,
    include_indices=True,
)