"""Utilities module for spherical tiling and heatmap visualizations.

This module provides utility functions for data processing, spherical tile center and boundary calculations, heatmap calculations,
and visualization.
"""

from .data_utils import (
    generate_fibonacci_lattice,
    get_FB_tile_boundaries,
    get_ERP_tile_boundaries,
    get_CMP_tile_boundaries,
    vector_angle_distance,
    find_geodesic_distances,
    find_geodesic_distances_from_dict,
    find_ERP_distance,
    find_ERP_distances_from_dict,


    normalize_to_pixel,
    pixel_to_spherical,
    process_viewport_data,
    format_trajectory_data,
    validate_video_dimensions,


    normalize,
    find_perpendicular_on_tangent_plane,
    great_circle_intersection,
    get_line_segment,
    find_nearest_point,
    spherical_interpolation,


    get_tile_corners,
    triangulate_spherical_polygon,
    angle_at_vertex,
    calculate_spherical_triangle_area,
    compute_spherical_polygon_area,


    compute_FB_tile_areas,
    compute_ERP_tile_areas,
    compute_CMP_tile_areas,
)

from .heatmap_utils import (
    find_nearest_tile,
    calculate_tile_weights,
    compute_heatmap,
    HeatmapConfig,
    HeatFunction,
    HeatFunctionType,
)

from .visualization_utils import (
    save_fb_tiling_visualization_video,
    save_fb_tiling_visualization_image,
    save_tiling_visualization_video,
    save_tiling_visualization_image,

    weight_to_color,
    create_spherical_tile_patch,
    save_tiling_visualization_with_weights,
    plot_points_on_sphere,
    save_heatmap_ERP_image,
)

__all__ = [
    # Data utilities
    'generate_fibonacci_lattice',
    'get_FB_tile_boundaries',
    'get_ERP_tile_boundaries',
    'get_CMP_tile_boundaries',
    'vector_angle_distance',
    'find_geodesic_distances',
    'find_geodesic_distances_from_dict',
    'find_ERP_distance',
    'find_ERP_distances_from_dict',
    'normalize_to_pixel',
    'pixel_to_spherical',
    'process_viewport_data',
    'format_trajectory_data',
    'validate_video_dimensions',
    'normalize',
    'find_perpendicular_on_tangent_plane',
    'great_circle_intersection',
    'get_line_segment',
    'find_nearest_point',
    'spherical_interpolation',
    'get_tile_corners',
    'triangulate_spherical_polygon',
    'angle_at_vertex',
    'calculate_spherical_triangle_area',
    'compute_spherical_polygon_area',
    'compute_FB_tile_areas',
    'compute_ERP_tile_areas',
    'compute_CMP_tile_areas',
    
    # Heatmap utilities
    'find_nearest_tile',
    'calculate_tile_weights',
    'compute_heatmap'
    'HeatmapConfig',
    'HeatFunction',
    'HeatFunctionType',
    
    # Visualization utilities
    'save_fb_tiling_visualization_video',
    'save_fb_tiling_visualization_image',
    'save_tiling_visualization_video',
    'save_tiling_visualization_image',

    'weight_to_color',
    'create_spherical_tile_patch',
    'save_tiling_visualization_with_weights',
    'plot_points_on_sphere',
    'save_heatmap_ERP_image',
]