# fibonacci_lattice_tiling_toolkit

A Python package for visualizing Fibonacci lattice tiling of the sphere and generating heatmaps of 360-degree video viewers. This tool helps visualize how users' attention is distributed when watching 360-degree videos.

## Features

- Calculate Fibonacci lattice tiling tile centers and tile boundaries.
- Generate visualizations (.glb, images, video) for arbitrary tile boundaries.
- Generate visualizations (.glb, images, video) for heatmaps of arbitrary tiling.
- Configurable tiling and heatmap options.
- Comprehensive error handling and validation.

## Installation

### Requirements

- Python 3.8 or higher
- numpy
- pandas
- matplotlib
- ffmpeg (for video generation)

### Installation Steps

1. Clone the repository:
```bash
git clone 
cd fibonacci_lattice_tiling_toolkit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

### Headless Environments (e.g., Google Colab, servers)

If you are running this package in a headless environment and using features that require 3D rendering, you will need to install XVFB:

```bash
sudo apt-get install -y xvfb
```

## Quick Start

Here's a simple example to get you started:

```python
from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_fb_tiling_visualization_image, save_fb_tiling_visualization_video, save_fb_tiling_visualization_glb
from pathlib import Path


output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

# Generates Fibonacci lattice tiling with 29 tiles
save_fb_tiling_visualization_image(
    29,
    output_dir=output_dir
    )

save_fb_tiling_visualization_video(
    30,
    output_dir=output_dir,
    horizontal_pan=True,
    vertical_pan=True
    )

save_fb_tiling_visualization_glb(
    30,
    output_dir=output_dir
    )
```

## Usage

### Basic Usage

A basic way to generate tiling visualizations with default configurations:

```python
from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_fb_tiling_visualization_image
from pathlib import Path


output_dir = Path("./output")
# Ensure the directory exists
output_dir.mkdir(parents=True, exist_ok=True)

save_fb_tiling_visualization_image(
    29,
    output_dir=output_dir)
```

A basic way to generate heatmap visualizations on any data with default configurations:

```python
import os
from pathlib import Path
import random
import math
import csv

from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_fb_tiling_visualization_video, spherical_interpolation, save_tiling_visualization_with_weights
from fibonacci_lattice_tiling_toolkit.utilities.data_utils import generate_fibonacci_lattice, get_FB_tile_boundaries, get_ERP_tile_boundaries, get_CMP_tile_boundaries, get_CMP_tile_centers, Vector, vector_angle_distance, find_angular_distances
from fibonacci_lattice_tiling_toolkit.utilities.heatmap_utils import HeatmapConfig, calculate_tile_weights, calculate_tile_weights_by_index

# Output directory
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

input_dir = Path("./data/Examples")
input_file_name = "viewer_vectors-Test_Two_Clusters.csv"
input_vector_file = os.path.join(input_dir, input_file_name)

data_title = "Test_Two_Clusters"

viewer_vectors = {}

with open(input_vector_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = str(row["key"])
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        viewer_vectors[key] = Vector(x, y, z)

# Heatmap config is defaulted, using Geodesic distance and Gaussian smoothing heat function.
heatmap_config = HeatmapConfig()

# Generate tiling heatmap for Fibonacci lattice tiling on sphere.

tile_centers = generate_fibonacci_lattice(647)
tile_boundaries = get_FB_tile_boundaries(647)

output_prefix = f"FB_({len(tile_centers)})_tiling-{data_title}-"

tile_centers_dict = {}
for i in range(len(tile_centers)):
  tile_centers_dict[str(i)] = tile_centers[i]

tile_centers = tile_centers_dict

tile_weights = {}

for tile_index in tile_centers:
  tile_weights[tile_index] = 0.0

for tile_index in list(tile_boundaries.keys()):
  tile_boundaries[str(tile_index)] = tile_boundaries[tile_index]
  del tile_boundaries[tile_index]

for viewer in viewer_vectors.keys():
  viewer_vector = viewer_vectors[viewer]

  tile_weights_list = calculate_tile_weights_by_index(
      vector=viewer_vector,
      tile_centers=tile_centers,
      config=heatmap_config)

  for tile_index in tile_weights_list.keys():
    tile = tile_centers[tile_index]
    tile_weight = tile_weights_list[tile_index]

    tile_weights[tile_index] += tile_weight


save_tiling_visualization_with_weights(
    tile_boundaries=tile_boundaries,
    tile_weights=tile_weights,
    output_dir=output_dir,
    output_prefix=output_prefix)
```

### Area Calculation

To calculate the areas of an arbitrary tiling, here is an example:
```python
from fibonacci_lattice_tiling_toolkit.utilities.data_utils import compute_FB_tile_areas, compute_ERP_tile_areas, compute_CMP_tile_areas, get_FB_tile_boundaries
from fibonacci_lattice_tiling_toolkit.data_types import Vector
import matplotlib.pyplot as plt
import numpy as np
import csv

# Choose values for ERP list. These will be adjusted by FB and CMP to closest match the number of tiles.
ERP_tile_list = [(10, 5), (16, 8), (22, 11), (30, 15), (40, 20), (60, 30), (80, 40)]

tile_counts = []

for i, j in ERP_tile_list:
    total_tiles = i * j
    tile_counts.append(total_tiles)


for tile_count in tile_counts:
  adjusted_tile_count = int(tile_count)

  if (adjusted_tile_count % 2 == 0):
    adjusted_tile_count -= 1

  tile_area_dict, fraction_of_sphere_dict = compute_FB_tile_areas(adjusted_tile_count)

  print(tile_area_dict)

  print()

  print(fraction_of_sphere_dict)
  print(np.sum(list(fraction_of_sphere_dict.values())))

  print()

  # Compute mean tile area
  mean_tile_area = np.mean(list(tile_area_dict.values()))

  # Compute standard deviation of tile areas
  std_tile_area = np.std(list(tile_area_dict.values()))

  # Compute mean fraction of sphere
  mean_fraction = np.mean(list(fraction_of_sphere_dict.values()))

  # Compute standard deviation of fractions
  std_fraction = np.std(list(fraction_of_sphere_dict.values()))

  # Extract fraction values
  fractions = list(fraction_of_sphere_dict.values())

  # Compute quartiles and range
  q1 = np.percentile(fractions, 25)   # First quartile (Q1)
  median = np.percentile(fractions, 50)  # Median (Q2)
  q3 = np.percentile(fractions, 75)   # Third quartile (Q3)
  min_val = np.min(fractions)   # Minimum value
  max_val = np.max(fractions)   # Maximum value

  print(f"Mean tile area for fibonacci lattice tiling: {mean_tile_area}")
  print(f"Standard deviation of tile area for fibonacci lattice tiling: {std_tile_area}")

  print(f"Mean fraction of sphere for fibonacci lattice tiling: {mean_fraction}")
  print(f"Standard deviation of fraction for fibonacci lattice tiling: {std_fraction}")

  # Compute %RSD for tile areas
  rsd_tile_area = (std_tile_area / mean_tile_area) * 100 if mean_tile_area != 0 else 0

  # Compute %RSD for fraction of the sphere
  rsd_fraction = (std_fraction / mean_fraction) * 100 if mean_fraction != 0 else 0

  print(f"Relative Standard Deviation (RSD) of tile areas for fibonacci lattice tiling: {rsd_tile_area:.2f}%")
  print(f"Relative Standard Deviation (RSD) of sphere fractions for fibonacci lattice tiling: {rsd_fraction:.2f}%")
```

### Data Preprocessing

To Preprocess data, here is an example:

```python
import math
import csv
from fibonacci_lattice_tiling_toolkit.utilities.data_utils import Vector

def random_perturbation_on_sphere(v0, max_angle_deg):
    """Perturb vector v0 by a random direction within max_angle_deg."""
    max_angle_rad = math.radians(max_angle_deg)

    # Uniformly sample angle θ within spherical cap
    cos_theta = 1 - random.random() * (1 - math.cos(max_angle_rad))
    theta = math.acos(cos_theta)
    phi = random.uniform(0, 2 * math.pi)

    # Local coordinates: z is original direction
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    # Create local vector
    local_vec = np.array([x, y, z])

    # Create a rotation matrix to align +Z with v0
    z_axis = np.array([0, 0, 1])
    v0_np = np.array([v0.x, v0.y, v0.z])
    v0_norm = v0_np / np.linalg.norm(v0_np)

    axis = np.cross(z_axis, v0_norm)
    angle = math.acos(np.clip(np.dot(z_axis, v0_norm), -1, 1))

    if np.linalg.norm(axis) < 1e-8:
        # v0 is (anti)parallel to z_axis
        if np.dot(z_axis, v0_norm) > 0:
            rot_matrix = np.eye(3)
        else:
            rot_matrix = -np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rot_matrix = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    # Rotate local vector into global orientation
    global_vec = rot_matrix @ local_vec
    global_vec /= np.linalg.norm(global_vec)
    return Vector(global_vec[0], global_vec[1], global_vec[2])

viewer_vectors = {}

max_angular_dist = 5

# First cluster near (0°, 0°)
center_vec1 = Vector.from_spherical(lat=0, lon=0)
for i in range(15):
    viewer_vectors[i] = random_perturbation_on_sphere(center_vec1, max_angular_dist)

# Second cluster near (50°, 50°)
center_vec2 = Vector.from_spherical(lat=50, lon=50)
for i in range(15, 30):
    viewer_vectors[i] = random_perturbation_on_sphere(center_vec2, max_angular_dist)

print(viewer_vectors)

# Output directory
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = os.path.join(output_dir, f'viewer_vectors-Test_Two_Clusters.csv')

try:
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["key", "x", "y", "z"])

        # One row per vector
        for i in viewer_vectors:
            vec = viewer_vectors[i]
            writer.writerow([i, vec.x, vec.y, vec.z])

    print(f"Viewer vectors saved to: {output_file}")
except Exception as e:
    print(f"Error saving viewer vectors: {e}")
```

### Custom Configuration

Customize the analysis parameters:

```python
from fibonacci_lattice_tiling_toolkit.config import HeatmapConfig

# Create custom configuration
config = HeatmapConfig(
    fov_angle=100.0,
    use_erroneous_ERP_distance=False,
    heat_function=ExponentialHeatFunction()
)
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Installation Guide](docs/installation.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Example Scripts](examples/README.md)

## Project Structure

```
fibonacci_lattice_tiling_toolkit/
├── src/      # Main package
│   └── utilities/        # Utility functions
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example scripts
└── data/                 # Sample datasets
```

## Input Data Format

Vector: This data class represents a 3-dimensional vector, where x is the front-back axis,
        y is the left-right axis, and z is the up-down axis.

RadialPoint: This data class represents a spherical coordinate, where lat is the latitudinal coordinate,
             and lon is the longitudinal coordinate.


The visualization functions expects tiling_boundaries dictionary defined as follows:
- key (str): The key may be any string that uniquely defines the tile.
- value (list[list[Vector]]): The value must be a list of boundaries, where each boundary is a list of length 2,
    with the first is a Vector representing one point on the sphere at the start of the boundary, and the second
    is a Vector representing the other point at the end of the boundary.
The visualization functions expect tile_centers dictionary defined as follows:
- key (str): The key may be any string that uniquely defines the tile.
- value (Vector): The value must be the Vector that represents the center of the tile.


## Output

The analyzer generates:
1. .glb files for arbitrary tiling and heatmap visualizations on sphere.
2. Image and panning video files for aribtrary tiling and heatmap visualizations on sphere.
3. Image files for ERP tiling and heatmap visualizations on ERP image.
4. Image files for CMP tiling and heatmap visualizations on CMP image.

## Configuration Options

Key configuration parameters:

```python
AnalyzerConfig(
    video_width=100,         # Video width in pixels
    video_height=200,        # Video height in pixels
    tile_counts=[20, 50],    # Number of tiles for analysis
)

HeatmapConfig(
    fov_angle=120.0        # Field of view angle
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{fibonacci_lattice_tiling_toolkit,
  title = {Fibonacci Lattice Tiling},
  author = {Chitsein Htun},
  year = {2025},
  url = {https://github.com/Chtun/fibonacci_lattice_tiling_toolkit}
}
```

## Contact

For questions and support:
- Create an issue on GitHub
- Email: chtun@live.com

## Acknowledgments

This project is based on research from the University of Texas at Dallas and builds upon the concepts presented in "Addressing Non-Uniform Tiling of Visual Field in VR".
