"""Visualization utilities module for spherical tiling and heatmap visualization.

This module provides utilities for creating visualizations of spherical tiling and heatmap visualization
results, including heatmaps, trajectory plots, and animations.

Classes:
    VisualizationConfig: Configuration for visualization parameters.
    PlotManager: Manages plot creation and updates.

Functions:
    setup_plot: Configures matplotlib plot with given parameters.
    create_animation: Creates animation from trajectory data.
    save_video: Saves animation as video file.
    generate_color_map: Generates colors for heatmap visualization.

    save_fb_tiling_visualization_glb: Saves a Fibonacci lattice tiling on sphere as a .glb.
    save_fb_tiling_visualization_image: Saves a Fibonacci lattice tiling on sphere as an image.
    save_fb_tiling_visualization_video: Saves a Fibonacci lattice tiling on sphere as a video.
    save_tiling_visualization_glb: Saves an arbitrary tiling on sphere as a .glb.
    save_tiling_visualization_image: Saves an arbitrary tiling on sphere as an image.
    save_tiling_visualization_video: Saves an arbitrary tiling on sphere as a video.

    weight_to_color: Maps a weight to a color.
    create_spherical_tile_patch: Generates an pyvista PolyData object for a spherical patch.
    save_tiling_visualization_with_weights: Saves an arbitrary tiling on sphere with weights as a .glb.
    plot_points_on_sphere: Plots a set of Vector points on the sphere.
    save_heatmap_ERP_image: Saves an ERP heatmap based on tiling weights on an ERP image.
    save_CMP_heatmap_image: Saves a CMP heatmap based on tiling weights on a CMP image.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from dataclasses import dataclass

from fibonacci_lattice_tiling_toolkit import Vector, RadialPoint, ValidationError, convert_vectors_to_coordinates
from .data_utils import generate_fibonacci_lattice, spherical_interpolation, get_FB_tile_boundaries, get_tile_corners


def save_fb_tiling_visualization_glb(
        tile_count: int,
        output_dir: Path
        ):
    """Saves a video of the tiling on a sphere for fibonacci lattice.
    
    Args:
        tile_count: the number of Fibonacci lattice tiles to generate.
        output_dir: Path for output video folder.
    """

     # Grab tile center points and tile boundaries.
    tile_centers_vectors = generate_fibonacci_lattice(tile_count)
    tile_boundaries = get_FB_tile_boundaries(tile_count)

    # Convert spherical coordinates to Cartesian coordinates for plotting
    x = [vec.x for vec in tile_centers_vectors]
    y = [vec.y for vec in tile_centers_vectors]
    z = [vec.z for vec in tile_centers_vectors]

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    pv.start_xvfb()  # Start the virtual framebuffer

    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 50  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] #redefine line_segments.
    line_segment_count = 0 #keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) #add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) #create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) #create the points.
    lines.lines = np.array(line_segments).flatten() #define the lines.

    # Create PolyData for tile centers
    points = pv.PolyData(np.column_stack((x, y, z)))
    points['colors'] = np.array([[255, 0, 0]] * len(x))  # Red points

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)
    plotter.add_mesh(points, color='red', point_size=10) #plot tile centers

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    gltf_file_name = os.path.join(str(output_dir), f'fibonacci_lattice-{tile_count}_tiles.glb')

    try:
        plotter.export_gltf(gltf_file_name)  # This will export in GLTF format by default
        print(f"GLTF saved: {gltf_file_name}")
    except Exception as e:
        print(f"Error saving GLTF: {e}")

    plotter.close()

def save_fb_tiling_visualization_image(
        tile_count: int,
        output_dir: Path,
        camera_position: Tuple[float, float, float]=(0, 0, 5),
        camera_up: Tuple[float, float, float]= (0, 1, 0),
        camera_focal_point: Tuple[float, float, float] = (0,0,0),
        tile_center_size: float = 0.03,
        include_indices: bool = False,
        ):
    """Saves a video of the tiling on a sphere for fibonacci lattice.
    
    Args:
        tile_count: the number of Fibonacci lattice tiles to generate.
        output_dir: Path for output video folder.
        camera_position: The camera position.
        camera_up: The camera "up" vector.
        camera_focal_point: The focal point to focus on.
        include_indices: True if indices should be printed on tile centers, False otherwise.
    """

     # Grab tile center points and tile boundaries.
    tile_centers_vectors = generate_fibonacci_lattice(tile_count)
    tile_boundaries = get_FB_tile_boundaries(tile_count)

    # Convert spherical coordinates to Cartesian coordinates for plotting
    x = [vec.x for vec in tile_centers_vectors]
    y = [vec.y for vec in tile_centers_vectors]
    z = [vec.z for vec in tile_centers_vectors]

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
        pv.start_xvfb()


    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 500  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] # redefine line_segments.
    line_segment_count = 0 # keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) # add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) # create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) # create the points.
    lines.lines = np.array(line_segments).flatten() # define the lines.

     # Create PolyData for tile centers
    centers = np.column_stack((x, y, z))
    points = pv.PolyData(centers)

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)

     # Add larger markers for centers (spheres instead of just points)
    for idx, (cx, cy, cz) in enumerate(centers):
        plotter.add_mesh(pv.Sphere(radius=tile_center_size, center=(cx, cy, cz)), color='red')

    # Add labels (tile indices)
    if include_indices:
        labels = [str(i) for i in range(tile_count)]
        plotter.add_point_labels(points, labels, point_size=20, text_color="blue", font_size=12)

    # Set camera view
    plotter.camera.position = camera_position
    plotter.camera.up = camera_up
    plotter.camera.focal_point = camera_focal_point

    file_name_suffix = f"-camera_position_{camera_position[0]}_{camera_position[1]}_{camera_position[2]}-camera_up_{camera_up[0]}_{camera_up[1]}_{camera_up[2]}"

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    # Create filename
    png_file_name = os.path.join(str(output_dir), f'fibonacci_lattice-{tile_count}_tiles{file_name_suffix}.png')

    try:
        plotter.screenshot(png_file_name)
        print(f"PNG saved: {png_file_name}")
    except Exception as e:
        print(f"Error saving PNG: {e}")

    plotter.close()

def save_fb_tiling_visualization_video(tile_count: int, output_dir: Path, horizontal_pan: bool=True, vertical_pan: bool=True):
    """Saves a video of the tiling on a sphere for fibonacci lattice.
    
    Args:
        tile_count: the number of Fibonacci lattice tiles to generate.
        output_dir: Path for output video folder.
        horizontal_pan: Whetehr to horizontally pan on the video.
        vertical_pan: Whether to vertically pan on the video.
    """

    if (horizontal_pan is False and vertical_pan is False):
        raise ValidationError("Video must pan horizontally or vertically or both!")

     # Grab tile center points and tile boundaries.
    tile_centers_vectors = generate_fibonacci_lattice(tile_count)
    tile_boundaries = get_FB_tile_boundaries(tile_count)

    # Convert spherical coordinates to Cartesian coordinates for plotting
    x = [vec.x for vec in tile_centers_vectors]
    y = [vec.y for vec in tile_centers_vectors]
    z = [vec.z for vec in tile_centers_vectors]

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    pv.start_xvfb()  # Start the virtual framebuffer

    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 50  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] #redefine line_segments.
    line_segment_count = 0 #keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) #add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) #create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) #create the points.
    lines.lines = np.array(line_segments).flatten() #define the lines.

    # Create PolyData for tile centers
    points = pv.PolyData(np.column_stack((x, y, z)))
    points['colors'] = np.array([[255, 0, 0]] * len(x))  # Red points

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)
    plotter.add_mesh(points, color='red', point_size=10) #plot tile centers

    # Set view and remove axes
    plotter.camera.position = (0, 0, 5) #adjust camera location.
    plotter.camera.up = (0, 1, 0)
    plotter.camera.focal_point = (0,0,0)

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    horizontal_pan_val = 0
    vertical_pan_val = 0
    file_name_suffix = ""

    if (horizontal_pan and vertical_pan):
        horizontal_pan_val = 0.5
        vertical_pan_val = 0.5
        file_name_suffix = "-vertical_horizontal"
    elif (horizontal_pan):
        horizontal_pan_val = 1
        file_name_suffix = "-horizontal"
    elif (vertical_pan):
        vertical_pan_val = 1
        file_name_suffix = "-vertical"

    try:
        # Open movie file
        video_file_name = os.path.join(str(output_dir), f'fibonacci_lattice-{tile_count}_tiles{file_name_suffix}.mp4')
        plotter.open_movie(video_file_name)

        # Rotate camera and write frames
        for i in range(180):
            plotter.camera.azimuth += horizontal_pan_val  # Rotate camera 1 degree
            plotter.camera.elevation += vertical_pan_val
            plotter.write_frame()
        
        print(f"Video saved: {video_file_name}")
    except Exception as e:
        print(f"Error saving video: {e}")

    plotter.close()

def save_tiling_visualization_glb(
        tile_boundaries: Dict[int, List[List[Vector]]],
        output_dir: Path,
        output_prefix: str=""
        ):
    """Saves a .glb of the tiling on a sphere for fibonacci lattice.
    
    Args:
        tile_boundaries: The boundaries for the tiles to generate.
        output_dir: Path for output .glb folder.
    """

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    pv.start_xvfb()  # Start the virtual framebuffer

    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 50  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] #redefine line_segments.
    line_segment_count = 0 #keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) #add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) #create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) #create the points.
    lines.lines = np.array(line_segments).flatten() #define the lines.

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    gltf_file_name = os.path.join(str(output_dir), f'{output_prefix}tiling_visualization.glb')

    try:
        plotter.export_gltf(gltf_file_name)  # This will export in GLTF format by default
        print(f"GLTF saved: {gltf_file_name}")
    except Exception as e:
        print(f"Error saving GLTF: {e}")

def save_tiling_visualization_image(
        tile_boundaries: Dict[int, List[List[Vector]]],
        output_dir: Path,
        output_prefix: str="",
        camera_position: Tuple[float, float, float]=(0, 0, 5),
        camera_up: Tuple[float, float, float]= (0, 1, 0),
        camera_focal_point: Tuple[float, float, float] = (0,0,0),
        camera_azimuth: int = 0,
        camera_elevation: int = 0
        ):
    """Saves a video of the tiling on a sphere for fibonacci lattice.
    
    Args:
        tile_boundaries: The boundaries for the tiles to generate.
        output_dir: Path for output video folder.
        camera_position: The camera position.
        camera_up: The camera "up" vector.
        camera_focal_point: The focal point to focus on.
        camera_azimuth: The azimuth angle to set the camera to.
        camera_elevation: int = The elevation angle to set the camera to.
    """

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    pv.start_xvfb()  # Start the virtual framebuffer

    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 50  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] #redefine line_segments.
    line_segment_count = 0 #keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) #add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) #create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) #create the points.
    lines.lines = np.array(line_segments).flatten() #define the lines.

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)

    # Set camera view
    plotter.camera.position = camera_position
    plotter.camera.up = camera_up
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.azimuth = camera_azimuth
    plotter.camera.elevation = camera_elevation

    file_name_suffix = f"-camera_position_{camera_position[0]}_{camera_position[1]}_{camera_position[2]}-camera_up_{camera_up[0]}_{camera_up[1]}_{camera_up[2]}-azimuth_{camera_azimuth}-elevation_{camera_elevation}"

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    # Create filename
    png_file_name = os.path.join(str(output_dir), f'{output_prefix}tiling_visualization{file_name_suffix}.png')

    try:
        plotter.screenshot(png_file_name)
        print(f"PNG saved: {png_file_name}")
    except Exception as e:
        print(f"Error saving PNG: {e}")

    plotter.close()
    
def save_tiling_visualization_video(
        tile_boundaries: Dict[int, List[List[Vector]]],
        output_dir: Path,
        output_prefix: str="",
        horizontal_pan: bool=True,
        vertical_pan: bool=True,
        camera_position: Tuple[float, float, float]=(0, 0, 5),
        camera_up: Tuple[float, float, float]= (0, 1, 0),
        camera_focal_point: Tuple[float, float, float] = (0,0,0)
        ):
    """Saves a video of the tiling on a sphere for fibonacci lattice.
    
    Args:
        tile_boundaries: The boundaries for the tiles to generate.
        output_dir: Path for output video folder.
        horizontal_pan: Whetehr to horizontally pan on the video.
        vertical_pan: Whether to vertically pan on the video.
        camera_position: The camera position.
        camera_up: The camera "up" vector.
        camera_focal_point: The focal point to focus on.
    """

    if (horizontal_pan is False and vertical_pan is False):
        raise ValidationError("Video must pan horizontally or vertically or both!")

    # Extract lines for tile boundary
    tile_boundary_list = []
    for boundaries in tile_boundaries.values():
        for boundary in boundaries:
            tile_boundary_list.append(boundary)

    pv.start_xvfb()  # Start the virtual framebuffer

    sphere_radius = 1 # Change the radius here
    sphere_opacity = 0.3  # Change the opacity here
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    # Generate points for each arc in the boundaries
    num_points = 50  # Number of points on the arc
    t_values = np.linspace(0, 1, num_points)

    arc_points_list = []
    line_segments = [] #redefine line_segments.
    line_segment_count = 0 #keep track of segment number.

    for boundary in tile_boundary_list:
        vec_start, vec_end = boundary
        arc_points = np.array([spherical_interpolation(vec_start, vec_end, t) for t in t_values])
        for i in range(len(arc_points) - 1):
            arc_points_list.extend(arc_points[i:i+2]) #add the two points of the segment.
            line_segments.append([2, line_segment_count*2, line_segment_count*2+1]) #create a line segment.
            line_segment_count +=1

    # Create PolyData for lines
    lines = pv.PolyData(np.array(arc_points_list)) #create the points.
    lines.lines = np.array(line_segments).flatten() #define the lines.

    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(sphere)
    plotter.add_mesh(lines, color='black', line_width=2)

    # Set view and remove axes
    plotter.camera.position = camera_position
    plotter.camera.up = camera_up
    plotter.camera.focal_point = camera_focal_point

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    horizontal_pan_val = 0
    vertical_pan_val = 0
    file_name_suffix = ""

    if (horizontal_pan and vertical_pan):
        horizontal_pan_val = 0.5
        vertical_pan_val = 0.5
        file_name_suffix = "-vertical_horizontal"
    elif (horizontal_pan):
        horizontal_pan_val = 1
        file_name_suffix = "-horizontal"
    elif (vertical_pan):
        vertical_pan_val = 1
        file_name_suffix = "-vertical"

    try:
        # Open movie file
        video_file_name = os.path.join(str(output_dir), f'{output_prefix}tiling_visualization{file_name_suffix}.mp4')
        plotter.open_movie(video_file_name)

        # Rotate camera and write frames
        for i in range(180):
            plotter.camera.azimuth += horizontal_pan_val  # Rotate camera 1 degree
            plotter.camera.elevation += vertical_pan_val
            plotter.write_frame()
        
        print(f"Video saved: {video_file_name}")
    except Exception as e:
        print(f"Error saving PNG: {e}")

    plotter.close()


def weight_to_color(weight: float, min_weight: float, max_weight: float) -> Tuple[float, float, float]:
    """Maps a weight to a color."""
    if max_weight - min_weight > 0:
      normalized_weight = (weight - min_weight) / (max_weight - min_weight)
    else:
      normalized_weight = 1
    colormap = plt.cm.inferno
    color = colormap(normalized_weight)
    return (color[0], color[1], color[2])

def create_spherical_tile_patch(tile_corners: List[Vector], sphere_radius=1.0, steps=50) -> pv.PolyData:
    arc_points = []

    # Interpolate between each pair of consecutive corner points
    for i in range(len(tile_corners)):
        start = tile_corners[i]
        end = tile_corners[(i + 1) % len(tile_corners)]  # Wrap around to close the loop
        arc = [spherical_interpolation(start, end, t) for t in np.linspace(0, 1, steps)]
        arc_points.extend(arc)

    # Normalize the arc points to be on the sphere surface
    normalized_arc_points = np.array([sphere_radius * np.array(v) / np.linalg.norm(v) for v in arc_points])

    # Calculate the center point (average of the corners, then normalized)
    center = np.mean(arc_points, axis=0)
    center_normalized = center / np.linalg.norm(center)
    center_point = center_normalized * sphere_radius

    # Include the center as the first point
    points = [center_point] + normalized_arc_points.tolist()

    faces = []
    num_boundary_points = len(normalized_arc_points)

    # Create faces connecting the center to consecutive boundary points
    for i in range(num_boundary_points):
        # The center is always index 0
        p1_index = i + 1  # Index of the current boundary point
        p2_index = (i + 1) % num_boundary_points + 1  # Index of the next boundary point (wraps around)
        
        faces.append([3, 0, p1_index, p2_index])

    # Flatten the faces array
    faces = np.hstack(faces)

    patch = pv.PolyData(np.array(points), faces)

    return patch

# Assuming you have your patch creation function like `create_spherical_tile_patch`
def make_double_sided(patch):
    # Flip the normals of the patch to create the back side
    flipped_patch = patch.copy()
    flipped_patch.flip_normals()
    return flipped_patch

def save_tiling_visualization_with_weights(
        tile_boundaries: Dict[str, List[List[Tuple[float, float, float]]]],
        tile_weights: Dict[str, float],
        output_dir: Path,
        steps=500,
        output_prefix: str=""
        ):
    """Generates a .glb of the tiling visualization with weights (heatmap) applied to each tile.
    
    Args:
        tile_boundaries: The boundaries for each tile (key is unique identifier for the tile).
        tile_weights: The weights for each tile (key is unique identifier for the tile).
        output_dir: Path for output video folder.
        step: The number of lines to draw for each patch (higher gives better meshes).
        output_prefix: The prefix before the standard file name.
    """
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)

    sphere_radius = 1
    sphere_opacity = 0.3
    sphere = pv.Sphere(radius=sphere_radius)
    sphere.opacity = sphere_opacity
    sphere.color = 'grey'

    min_weight = min(tile_weights.values())
    max_weight = max(tile_weights.values())

    for tile_index, boundaries in tile_boundaries.items():

        tile_corners = get_tile_corners(boundaries)
        patch = create_spherical_tile_patch(tile_corners, sphere_radius=1.0, steps=steps)
        
        weight = tile_weights[tile_index]
        color = weight_to_color(weight, min_weight, max_weight)

        # Make the mesh double-sided
        double_sided_patch = make_double_sided(patch)

        # Add both the original and flipped mesh to the plotter
        plotter.add_mesh(patch, color=color, opacity=1.0, lighting=False)
        plotter.add_mesh(double_sided_patch, color=color, opacity=1.0, lighting=False)

    plotter.enable_parallel_projection()
    plotter.show_axes_all()
    plotter.remove_bounds_axes()

    gltf_file_name = os.path.join(str(output_dir), f'{output_prefix}tiling_visualization_with_weights.glb')

    try:
        plotter.export_gltf(gltf_file_name)  # This will export in GLTF format by default
        print(f"GLTF saved: {gltf_file_name}")
    except Exception as e:
        print(f"Error saving GLTF: {e}")

    plotter.close()

def plot_points_on_sphere(
    vectors: Dict[str, Vector],
    output_dir: Path,
    output_prefix: str = ""
):
    """
    Plots a black sphere with latitude and longitude grid lines and small red dots
    at the positions specified in `vectors`.

    Args:
        vectors: Dictionary where keys are vector identifiers and values are Vectors
        representing a point on the sphere.
        output_dir: The path to the folder to save the file to.
        output_prefix: A prefix to place before the standard file name.
    """
    import pyvista as pv
    import numpy as np
    import os

    pv.start_xvfb()
    plotter = pv.Plotter()

    # Create the black sphere with higher resolution
    sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
    plotter.add_mesh(sphere, color='black', opacity=1.0)

    # Add latitude and longitude grid lines
    def add_lat_long_grid(plotter, radius=1.0, lat_step=15, lon_step=15):
        # Latitude lines (horizontal)
        for lat_deg in range(-90 + lat_step, 90, lat_step):
            lat_rad = np.radians(lat_deg)
            points = []
            for lon_deg in range(0, 361, 5):
                lon_rad = np.radians(lon_deg)
                x = radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = radius * np.cos(lat_rad) * np.sin(lon_rad)
                z = radius * np.sin(lat_rad)
                points.append([x, y, z])
            line = pv.lines_from_points(np.array(points), close=True)
            plotter.add_mesh(line, color="white", line_width=1)

        # Longitude lines (vertical)
        for lon_deg in range(0, 360, lon_step):
            lon_rad = np.radians(lon_deg)
            points = []
            for lat_deg in range(-90, 91, 5):
                lat_rad = np.radians(lat_deg)
                x = radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = radius * np.cos(lat_rad) * np.sin(lon_rad)
                z = radius * np.sin(lat_rad)
                points.append([x, y, z])
            line = pv.lines_from_points(np.array(points), close=False)
            plotter.add_mesh(line, color="white", line_width=1)

    add_lat_long_grid(plotter)

    # Plot small red dots at each viewer's position
    for idx, vector in vectors.items():
        pos = (vector.x, vector.y, vector.z)
        dot_sphere = pv.Sphere(radius=0.03, center=pos, theta_resolution=20, phi_resolution=20)
        plotter.add_mesh(dot_sphere, color='red')

    # Save the plot as a GLTF file
    gltf_file_name = os.path.join(str(output_dir), f'{output_prefix}point_visualization.glb')

    try:
        plotter.export_gltf(gltf_file_name)
        print(f"GLTF saved: {gltf_file_name}")
    except Exception as e:
        print(f"Error saving GLTF: {e}")

    plotter.close()

def save_heatmap_ERP_image(
    tile_weights: Dict[str, int],
    num_tiles_horizontal: int,
    num_tiles_vertical: int,
    output_dir: Path,
    output_prefix: str = "",
):
    """Save a heatmap of ERP tile weights as an image.
    
    Args:
        tile_weights: Dictionary where keys are tile_center identifiers and values are weights.
        num_tiles_horizontal: The number of tiles horizontally in the ERP.
        num_tiles_vertical: The number of tiles vertically in the ERP.
        output_dir: The path to the folder to save the file to.
        output_prefix: A prefix to place before the standard file name.
    """
    
    # Create a 2D array to store weights (j is vertical, i is horizontal)
    heatmap = np.zeros((num_tiles_vertical, num_tiles_horizontal))

    for tile_index, weight in tile_weights.items():
        try:
            j_str, i_str = tile_index.split("_")  # row (lat), col (lon)
            j = int(j_str)
            i = int(i_str)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid tile_index format: '{tile_index}'. Expected format 'j_i' with integers.") from e


        # Make sure indices are within bounds
        if 0 <= j < num_tiles_vertical and 0 <= i < num_tiles_horizontal:
            heatmap[j, i] = weight
        else:
            raise ValueError(f"Invalid tile_index: '{tile_index}'. Tile index was out of bounds.")

    # Normalize the heatmap for better visualization
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    # Flip vertically so that south pole is at the bottom
    heatmap = np.flipud(heatmap)

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    cax = ax.imshow(heatmap, cmap=plt.cm.inferno, interpolation='nearest',
                    extent=[-180, 180, -90, 90], aspect='auto')

    # Hide axes
    ax.axis('off')

    # Save the image
    filename = f"{output_prefix}heatmap.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"ERP heatmap saved to: {output_path}")

def save_CMP_heatmap_image(
    tile_weights: Dict[str, float],
    num_tiles_horizontal: int,
    num_tiles_vertical: int,
    output_dir: Path,
    output_prefix: str = "",
):
    """Creates and saves a CMP image using tile weights to generate a heatmap for each face.

    Args:
        tile_weights: Dictionary where keys are strings in the format "X=1-i_j" or "X=-1-i_j" and values are weights.
        num_tiles_horizontal: The number of tiles horizontally in the CMP.
        num_tiles_vertical: The number of tiles vertically in the CMP.
        output_dir: The path to the folder to save the file to.
        output_prefix: A prefix to place before the standard file name.
    """
    
    # Calculate the size of each face based on the number of tiles
    face_width = num_tiles_horizontal * 100  # 100 pixels per tile
    face_height = num_tiles_vertical * 100  # 100 pixels per tile
    
    # Create an empty image for the unfolded CMP grid (with transparent background)
    grid_width = face_width * 4  # 4 faces horizontally (left, front, right, back)
    grid_height = face_height * 3  # 3 faces vertically (top, left, bottom)
    image = np.zeros((grid_height, grid_width, 4), dtype=np.float32)  # clear RGBA image
    
    # Prepare heatmaps for each face
    heatmaps = {
        "top": np.zeros((num_tiles_vertical, num_tiles_horizontal)),
        "bottom": np.zeros((num_tiles_vertical, num_tiles_horizontal)),
        "left": np.zeros((num_tiles_vertical, num_tiles_horizontal)),
        "front": np.zeros((num_tiles_vertical, num_tiles_horizontal)),
        "right": np.zeros((num_tiles_vertical, num_tiles_horizontal)),
        "back": np.zeros((num_tiles_vertical, num_tiles_horizontal)),
    }

    # Map tile_weights to corresponding heatmap
    for tile_index, weight in tile_weights.items():
        try:
            # Split the tile_index at the last '-'
            plane_str, index_str = tile_index.rsplit('-', 1)
            
            # Parse the X part (1 for top, -1 for bottom)
            plane_val = int(plane_str.split('=')[1])
            plane = (plane_str.split('=')[0].split('_')[1])

            # Parse the i, j part (horizontal and vertical indices)
            i, j = map(int, index_str.split('_'))
        except ValueError:
            raise ValueError(f"Invalid tile_index format: '{tile_index}'. Expected format 'face_X=-1-i_j'.")

        # Determine which face the tile belongs to based on Plane
        if plane == "Z" and plane_val == 1:  # Top face
            heatmaps["top"][i, j] = weight
        elif plane == "Z" and plane_val == -1:  # Bottom face
            heatmaps["bottom"][i, j] = weight
        elif plane == "X" and plane_val == 1:
            heatmaps["front"][j, i] = weight
        elif plane == "X" and plane_val == -1:
            heatmaps["back"][j, i] = weight
        elif plane == "Y" and plane_val == 1:
            heatmaps["right"][j, i] = weight
        elif plane == "Y" and plane_val == -1:
            heatmaps["left"][j, i] = weight

    # Find the global maximum value across all heatmaps
    global_max_val = np.max([np.max(heatmaps[face]) for face in heatmaps])

    # Normalize all heatmaps by the global maximum value
    for face in heatmaps:
        if global_max_val > 0:
            heatmaps[face] = heatmaps[face] / global_max_val

    # Generate the heatmap colors for each face using a colormap
    cmap = plt.cm.inferno  # You can change this to any colormap you prefer
    top_face_color = cmap(heatmaps["top"])
    bottom_face_color = cmap(heatmaps["bottom"])
    left_face_color = cmap(heatmaps["left"])
    front_face_color = cmap(heatmaps["front"])
    right_face_color = cmap(heatmaps["right"])
    back_face_color = cmap(heatmaps["back"])

    # Define positions of the faces in the unfolded grid
    top_start = (0, face_width)  # Top face position
    bottom_start = (2 * face_height, face_width)  # Bottom face position
    left_start = (face_height, 0)  # Left face position
    front_start = (face_height, face_width)  # Front face position
    right_start = (face_height, 2 * face_width)  # Right face position
    back_start = (face_height, 3 * face_width)  # Back face position

    # Assign colors to each face in the unfolded grid
    face_positions = [
        (top_start, top_face_color),
        (bottom_start, bottom_face_color),
        (left_start, left_face_color),
        (front_start, front_face_color),
        (right_start, right_face_color),
        (back_start, back_face_color),
    ]
    
    # Resize heatmaps to fit the scale of each face in the unfolded grid
    for (y, x), color in face_positions:
        # Scale heatmap to match face size (num_tiles * 100)
        scaled_color = np.kron(color, np.ones((100, 100, 1)))  # Scale by 100x100
        image[y:y + face_height, x:x + face_width] = scaled_color

    # Rotate the front X face faces: Flip vertically and/or horizontally to line up the correct edges
    image[front_start[0]:front_start[0] + face_height, front_start[1]:front_start[1] + face_width] = np.flipud(image[front_start[0]:front_start[0] + face_height, front_start[1]:front_start[1] + face_width])
    image[back_start[0]:back_start[0] + face_height, back_start[1]:back_start[1] + face_width] = np.flipud(np.fliplr(image[back_start[0]:back_start[0] + face_height, back_start[1]:back_start[1] + face_width]))


    # Rotate the Y (right and left) faces: Flip vertically and/or horizontally to line up the correct edges
    image[right_start[0]:right_start[0] + face_height, right_start[1]:right_start[1] + face_width] = np.flipud(np.fliplr(image[right_start[0]:right_start[0] + face_height, right_start[1]:right_start[1] + face_width]))
    image[left_start[0]:left_start[0] + face_height, left_start[1]:left_start[1] + face_width] = np.flipud(image[left_start[0]:left_start[0] + face_height, left_start[1]:left_start[1] + face_width])



    # Rotate the bottom face: Flip the bottom face vertically to match the values
    image[bottom_start[0]:bottom_start[0] + face_height, bottom_start[1]:bottom_start[1] + face_width] = np.flipud(image[bottom_start[0]:bottom_start[0] + face_height, bottom_start[1]:bottom_start[1] + face_width])

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')

    # Save the image
    filename = f"{output_prefix}CMP_({num_tiles_horizontal}x{num_tiles_vertical})-CMP_heatmap.png"
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    print(f"CMP unfolded layout with tile weights saved to: {output_path}")