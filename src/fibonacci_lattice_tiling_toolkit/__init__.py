"""Spherical Tiling and Heatmap Visualization Package.

This package provides tools for generating and analyzing area of spherical tilings, as well as generating heatmaps for visualization on 360-degree video viewer data.
It includes functionality for data processing, spherical tile center and boundary calculations, heatmap calculations, and visualization.
"""

from .data_types import Point, RadialPoint, Vector, ValidationError, SpatialError, convert_vectors_to_coordinates
from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_fb_tiling_visualization_glb, save_tiling_visualization_glb,  save_CMP_heatmap_image, save_heatmap_ERP_image, save_tiling_visualization_with_weights
from fibonacci_lattice_tiling_toolkit.utilities.data_utils import get_FB_tile_boundaries, get_CMP_tile_boundaries, get_ERP_tile_boundaries, generate_fibonacci_lattice, get_CMP_tile_centers

__version__ = "1.0.0"
__author__ = "Chitsein Htun"
__all__ = [
    'Point',
    'RadialPoint',
    'Vector',
    'ValidationError',
    'SpatialError',
    'convert_vectors_to_coordinates',

    'HeatmapConfig',
    'save_fb_tiling_visualization_glb',
    'save_tiling_visualization_glb',
    'save_CMP_heatmap_image',
    'save_heatmap_ERP_image',
    'save_tiling_visualization_with_weights',

    'get_FB_tile_boundaries',
    'get_CMP_tile_boundaries',
    'get_ERP_tile_boundaries',
    'generate_fibonacci_lattice',
    'get_CMP_tile_centers',
]
