"""Data utilities module for spherical tiling analysis and heatmap visualizations.

This module provides utilities for data processing and coordinate transformations
in spherical tiling analysis and heatmap visualizations. It includes functions for
converting between different coordinate systems and processing viewport data.

Functions:
    generate_fibonacci_lattice: Generates uniformly distributed points on a sphere.
    get_FB_tile_boundaries: Generates the tiles (their boundaries) for Fibonacci-lattice Voronoi tiling.
    get_ERP_tile_boundaries: Generates the tiles (their boundaries) for ERP tiling.
    get_CMP_tile_boundaries: Generates the tiles (their boundaries) for CMP tiling.
    get_CMP_tile_centers: Generates the tile centers for CMP tiling.

    vector_angle_distance: Finds the angle between two Vectors.
    find_geodesic_distances: Finds the geodesic distance between a single Vector and a list of other Vectors.
    find_geodesic_distances_from_dict: Finds the geodesic distances between a single Vector and a dictionary of other Vectors.
    find_ERP_distance: Finds the Euclidean distance on an ERP between two Vectors.
    find_ERP_distances_from_dict: Finds the ERP distances between a single Vector and a dictionary of other Vectors.

    normalize_to_pixel: Converts normalized coordinates to pixel coordinates.
    pixel_to_spherical: Converts pixel coordinates to spherical coordinates.
    process_viewport_data: Processes viewport center trajectory data.
    validate_video_dimensions: Validates video dimensions.
    format_trajectory_data: Formats trajectory data for analysis.

    compute_FB_tile_areas: Computes the area for each FB tile.
    compute_ERP_tile_areas: Computes the area for each ERP tile.
    compute_CMP_tile_areas: Computes the area for each CMP tile.
    compute_spherical_polygon_area: Computes the area for a convex tile.
    calculate_spherical_triangle_area: Computes the area for a spherical triangle.
"""

from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
from pathlib import Path

from fibonacci_lattice_tiling_toolkit import Pixel, RadialPoint, Vector, ValidationError

def generate_fibonacci_lattice(num_points: int) -> List[Vector]:
    """Generates Fibonacci lattice points on a sphere.
    
    Args:
        num_points: Number of points to generate.
    
    Returns:
        List[Vector]: List of unit vectors representing lattice points.
    
    Raises:
        ValidationError: If number of points is invalid.
    """
    if num_points <= 0:
        raise ValidationError("Number of points must be positive")
        
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vectors = []
    
    if (num_points % 2 == 0):
        num_points -= 1
        print(f"Warning: The number of points must be odd, so Fibonacci lattice point generation will be done with n={num_points}")

    N = int(np.floor((num_points - 1) / 2))
    
    for i in range(-N, N + 1):
        lat = np.arcsin(2 * i / (2 * N + 1)) * 180 / np.pi
        lon = (i % phi) * 360 / phi
        
        # Normalize longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180
        
        # Convert to vector
        vector = Vector.from_spherical(lon, lat)
        vectors.append(vector)
    
    return vectors

def get_FB_tile_boundaries(tile_count: int, furthest_search_factor: float = 1.7) -> Dict[int, List[List[Vector]]]:
    """Generates the tiles (their boundaries) for Fibonacci-lattice Voronoi tiling.
    
    Args:
        tile_count: The number of tiles in the Fibonacci lattice.

    Returns:
        Dict: A dictionary where each tile index has a list of tile boundaries.

    Raises:
        ValidationError: Error if tile count is less than 1.
    """

    if (tile_count <= 0):
        raise ValidationError("Tile counts cannot be less than 1 for to visualize tiling!")

    # Grab tile center points
    tile_centers_vectors = generate_fibonacci_lattice(tile_count)
    tile_boundaries = {}

    for index_i in range(len(tile_centers_vectors)):
        tile_center_i = tile_centers_vectors[index_i]
        tile_boundaries[index_i] = []

        # This stores for each neighbor j, the great circle perpendicular to the vector between the current tile and neighbor j at the vector's midpoint.
        # Each great circle can be thought of as a candidate for the line that a tile boundary sits on.
        great_circle_vectors = {}
        neighbors = []

        # Extract all of the neighbors of the current tile, and the geodesic distance between the current tile center and each neighbor's center.
        for index_j in range(0, len(tile_centers_vectors)):
            if (index_j == index_i):
                continue

            tile_center_j = tile_centers_vectors[index_j]

            # Find the vector between the two tile centers
            line_segment = get_line_segment(tile_center_i, tile_center_j)
            line_vector = Vector(line_segment[0], line_segment[1], line_segment[2])

            # Compute the length of the line segment
            length = np.linalg.norm(line_segment)
            neighbors.append([index_j, length])

            # This plane is the great circle that is half way between the current tile and this neighbor, 
            # (i.e. the great circle perpendicular at the midpoint of tile i and j).
            # We store the normal vector of the plane of the great circle.
            # This plane passes through the center of the sphere and defines the great circle by its intersection with the sphere.
            great_circle_vectors[index_j] = line_vector

        # Sort the other FB lattice points by how close they are to the current tile.
        neighbors.sort(key=lambda x: x[1])
        smallest_distance = neighbors[0][1]

        # For each neighbor, compute the candidate tile boundaries.
        for index_j in range(len(neighbors)):
            neighbor_j = neighbors[index_j]
            if (neighbor_j[1] >= smallest_distance * furthest_search_factor):
                break

            tile_index_j = neighbor_j[0]
            tile_center_j = tile_centers_vectors[tile_index_j]
            great_circle_j = great_circle_vectors[tile_index_j]

            intersections = []

            for index_k in range(len(neighbors)):
                neighbor_k = neighbors[index_k]
                if (index_k == index_j):
                    continue

                # If the distance of the current neighbor is larger than 1.7, then that neighbor is too far to affect the tile boundary.
                if (neighbor_k[1] >= smallest_distance * furthest_search_factor):
                    break

                tile_index_k = neighbor_k[0]
                great_circle_k = great_circle_vectors[tile_index_k]

                # Find the points of intersection between the great circles representing the candidate lines that a tile boundary sits on.
                # These are the candidate points for corners of the tile.
                p1, p2 = great_circle_intersection(np.array([great_circle_j.x, great_circle_j.y, great_circle_j.z]), np.array([great_circle_k.x, great_circle_k.y, great_circle_k.z]))

                p1_vec = Vector(p1[0], p1[1], p1[2])
                p2_vec = Vector(p2[0], p2[1], p2[2])

                # Take the point that is closer to the tile center, the other point is on the opposite side of the sphere.
                intersection_vector = find_nearest_point(p1_vec, p2_vec, tile_center_i)


                intersection_i_seg = get_line_segment(tile_center_i, intersection_vector)
                length_i = np.linalg.norm(intersection_i_seg).round(4)

                # Add the point as a the candidate tile corner.
                intersections.append([intersection_vector, length_i, tile_index_k])

            # If there are fewer than 2 intersections, then this neighbor j does not make up a tile boundary with the current tile.
            if (len(intersections) < 2):
                continue

            # Sort the intersections by distance to the current tile.
            intersections.sort(key=lambda x: x[1])

            # Check if the two shortest intersection points are closer to the current tile than the intersection of their great circles.
            # If they are, then that means that the tile boundary is valid.
            # Otherwise, it means that neighbor j does not make up a tile boundary with the current tile. 
            shortest_intersection = intersections[0]
            second_shortest_intersection = intersections[1]

            tile_index_a = shortest_intersection[2]
            tile_index_b = second_shortest_intersection[2]

            great_circle_a = great_circle_vectors[tile_index_a]
            great_circle_b = great_circle_vectors[tile_index_b]

            p1, p2 = great_circle_intersection(np.array([great_circle_a.x, great_circle_a.y, great_circle_a.z]), np.array([great_circle_b.x, great_circle_b.y, great_circle_b.z]))

            p1_vec = Vector(p1[0], p1[1], p1[2])
            p2_vec = Vector(p2[0], p2[1], p2[2])

            intersection_vector = find_nearest_point(p1_vec, p2_vec, tile_center_i)

            intersection_i_seg = get_line_segment(tile_center_i, intersection_vector)
            length_intersection = np.linalg.norm(intersection_i_seg).round(4)
            midpoint_ij = np.array([(tile_center_i.x + tile_center_j.x) / 2, (tile_center_i.y + tile_center_j.y) / 2, (tile_center_i.z + tile_center_j.z) / 2])
            midpoint_ij = normalize(midpoint_ij)
            midpoint_ij_vec = Vector(midpoint_ij[0], midpoint_ij[1], midpoint_ij[2])
            midpoint_i_seg = get_line_segment(tile_center_i, midpoint_ij_vec)
            length_midpoint = np.linalg.norm(midpoint_i_seg).round(4)

            # If the intersection of great circles a and b is closer to i than the midpoint of i and j,
            # then these intersection points do not form a valid tile boundary.
            # If the intersection is further away, then it is a valid tile boundary.
            if (length_intersection > length_midpoint):
                tile_boundary = [shortest_intersection[0], second_shortest_intersection[0]]
                tile_boundaries[index_i].append(tile_boundary)

        # For each tile boundary in this list, make sure that any that are super close together are merged into one.
        boundary_i_index = 0
        while boundary_i_index < len(tile_boundaries[index_i]):
            boundary_i = tile_boundaries[index_i][boundary_i_index]
            corner_i_0 = boundary_i[0]
            corner_i_1 = boundary_i[1]

            boundary_j_index = boundary_i_index + 1
            while boundary_j_index < len(tile_boundaries[index_i]):
                boundary_j = tile_boundaries[index_i][boundary_j_index]
                corner_j_0 = boundary_j[0]
                corner_j_1 = boundary_j[1]

                # If they aren't equal but are very closer together, then set both to the first one.
                seg = get_line_segment(corner_i_0, corner_j_0)
                seg_length = np.linalg.norm(seg)
                if (corner_i_0 != corner_j_0) and (seg_length < 0.0005):
                    tile_boundaries[index_i][boundary_j_index][0] = corner_i_0

                seg = get_line_segment(corner_i_0, corner_j_1)
                seg_length = np.linalg.norm(seg)
                if (corner_i_0 != corner_j_1) and (seg_length < 0.0005):
                    tile_boundaries[index_i][boundary_j_index][1] = corner_i_0

                seg = get_line_segment(corner_i_1, corner_j_0)
                seg_length = np.linalg.norm(seg)
                if (corner_i_1 != corner_j_0) and (seg_length < 0.0005):
                    tile_boundaries[index_i][boundary_j_index][0] = corner_i_1
                
                seg = get_line_segment(corner_i_1, corner_j_1)
                seg_length = np.linalg.norm(seg)
                if (corner_i_1 != corner_j_1) and (seg_length < 0.0005):
                    tile_boundaries[index_i][boundary_j_index][1] = corner_i_1

                boundary_j_index += 1

            boundary_i_index += 1
    
    return tile_boundaries

def get_ERP_tile_boundaries(num_tiles_horizontal: int, num_tiles_vertical: int, radius: float = 1.0) -> Dict[str, List[List[Vector]]]:
    """
    Generates the tiles (their boundaries) for ERP tiling. Total number of tiles will be 
    
    Args:
        num_tiles_horizontal: The number of horizontal spatial bins (longitudes) to tile the ERP with.
        num_tiles_vertical: The number of vertical spatial bins (latitudes) to tile the ERP with.
        radius: the radius of the circle.

    Returns:
        Dict: A dictionary where each tile index "i_j" has a list of tile boundaries, where i is horizontal index and j is vertical index.

    Raises:
        ValueError: If either num_tiles_horizontal or num_tiles_vertical is not a positive integer.
    """

    if num_tiles_horizontal <= 0 or num_tiles_vertical <= 0:
        raise ValidationError("Number of tiles horizontal and vertical must be positive!")

    lat_step = 180 / num_tiles_vertical  # Latitude step size
    lon_step = 360 / num_tiles_horizontal  # Longitude step size

    ERP_tile_boundaries = {}

    # North and South poles
    north_pole = Vector(0, 0, radius)
    south_pole = Vector(0, 0, -radius)

    for i in range(num_tiles_vertical):
        lat1 = -90 + i * lat_step  # Bottom latitude
        lat2 = -90 + (i + 1) * lat_step  # Top latitude

        for j in range(num_tiles_horizontal):
            lon1 = -180 + j * lon_step  # Left longitude
            lon2 = -180 + (j + 1) * lon_step  # Right longitude

            # Convert to Cartesian coordinates
            p1 = Vector.from_spherical(lat=lat1, lon=lon1)  # Bottom-left
            p2 = Vector.from_spherical(lat=lat1, lon=lon2)  # Bottom-right
            p3 = Vector.from_spherical(lat=lat2, lon=lon2)  # Top-right
            p4 = Vector.from_spherical(lat=lat2, lon=lon1)  # Top-left

            index_key = f"{i}_{j}"

            if lat2 >= 90:  # North Pole Region (Triangular Tiles)
                ERP_tile_boundaries[index_key] = [[p1, p2], [p1, north_pole], [p2, north_pole]]  # Triangle with the pole
            elif lat1 <= -90:  # South Pole Region (Triangular Tiles)
                ERP_tile_boundaries[index_key] = [[p3, p4],[p3, south_pole], [p4, south_pole]]  # Triangle with the pole
            else:  # Normal Quadrilateral Tiles
                ERP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]

    return ERP_tile_boundaries

def get_CMP_tile_boundaries(num_tiles_horizontal: int, num_tiles_vertical: int, radius: float = 1.0) -> Dict[str, List[List[Vector]]]:
    """
    Generates the tiles (their boundaries) for CMP tiling. Total number of tiles will be 6 * num_tiles_horizontal * num_tiles_vertical.

    Args:
        num_tiles_horizontal: The number of horizontal spatial bins to tile a face of the CMP cube with.
        num_tiles_vertical: The number of vertical spatial bins to tile a face of the CMP cube with.
        radius: the radius of the circle.

    Returns:
        Dict: A dictionary where each tile index "face_FACE-i_j" has a list of tile boundaries, where FACE denotes which axis value is constant (e.g. Z=1)
         and i is horizontal index and j is vertical index.

    Raises:
        ValidationError: Error if tile count is less than 1.
    """

    if num_tiles_horizontal <= 0 or num_tiles_vertical <= 0:
        raise ValidationError("Number of tiles horizontal and vertical must be positive!")

    CMP_tile_boundaries = {}



    horizontal_step = 2 * radius / num_tiles_horizontal
    vertical_step = 2 * radius / num_tiles_vertical

    # Do the XY plane with Z = 1
    face_key = "Z=1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-radius + i * horizontal_step, -radius + j * vertical_step, radius)
            p2 = Vector(-radius + i * horizontal_step, -radius + (j + 1) * vertical_step, radius)
            p3 = Vector(-radius + (i + 1) * horizontal_step, -radius + (j + 1) * vertical_step, radius)
            p4 = Vector(-radius + (i + 1) * horizontal_step, -radius + j * vertical_step, radius)

            CMP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]


    # Do the XY plane with Z = -1
    face_key = "Z=-1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-radius + i * horizontal_step, -radius + j * vertical_step, -radius)
            p2 = Vector(-radius + i * horizontal_step, -radius + (j + 1) * vertical_step, -radius)
            p3 = Vector(-radius + (i + 1) * horizontal_step, -radius + (j + 1) * vertical_step, -radius)
            p4 = Vector(-radius + (i + 1) * horizontal_step, -radius + j * vertical_step, -radius)

            CMP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]

    # Do the XZ plane with Y = 1
    face_key = "Y=1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-radius + i * horizontal_step, radius, -radius + j * vertical_step)
            p2 = Vector(-radius + i * horizontal_step, radius, -radius + (j + 1) * vertical_step)
            p3 = Vector(-radius + (i + 1) * horizontal_step, radius, -radius + (j + 1) * vertical_step)
            p4 = Vector(-radius + (i + 1) * horizontal_step, radius, -radius + j * vertical_step)

            CMP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]

    # Do the XZ plane with Y = -1
    face_key = "Y=-1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-radius + i * horizontal_step, -radius, -radius + j * vertical_step)
            p2 = Vector(-radius + i * horizontal_step, -radius, -radius + (j + 1) * vertical_step)
            p3 = Vector(-radius + (i + 1) * horizontal_step, -radius, -radius + (j + 1) * vertical_step)
            p4 = Vector(-radius + (i + 1) * horizontal_step, -radius, -radius + j * vertical_step)

            CMP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]

    # Do the YZ plane with X = 1
    face_key = "X=1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(radius, -radius + i * horizontal_step, -radius + j * vertical_step)
            p2 = Vector(radius, -radius + i * horizontal_step, -radius + (j + 1) * vertical_step)
            p3 = Vector(radius, -radius + (i + 1) * horizontal_step, -radius + (j + 1) * vertical_step)
            p4 = Vector(radius, -radius + (i + 1) * horizontal_step, -radius + j * vertical_step)

            CMP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]

    # Do the YZ plane with X = -1
    face_key = "X=-1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-radius, -radius + i * horizontal_step, -radius + j * vertical_step)
            p2 = Vector(-radius, -radius + i * horizontal_step, -radius + (j + 1) * vertical_step)
            p3 = Vector(-radius, -radius + (i + 1) * horizontal_step, -radius + (j + 1) * vertical_step)
            p4 = Vector(-radius, -radius + (i + 1) * horizontal_step, -radius + j * vertical_step)

            CMP_tile_boundaries[index_key] = [[p1, p2], [p1, p4], [p2, p3], [p3, p4]]


    return CMP_tile_boundaries

def get_CMP_tile_centers(num_tiles_horizontal: int, num_tiles_vertical: int, radius: float = 1.0) -> Dict[str, Vector]:
    """
    Computes the center points of CMP tiles. Each tile center is the average of the 4 corners of the tile.

    Args:
        num_tiles_horizontal: The number of horizontal spatial bins to tile a face of the CMP cube with.
        num_tiles_vertical: The number of vertical spatial bins to tile a face of the CMP cube with.
        radius: the radius of the circle.

    Returns:
        Dict: A dictionary where each tile index "face_FACE-i_j" has a center point (Vector).
    """
    
    if num_tiles_horizontal <= 0 or num_tiles_vertical <= 0:
        raise ValidationError("Number of tiles horizontal and vertical must be positive!")
    
    CMP_tile_centers = {}

    horizontal_step = 2 / num_tiles_horizontal
    vertical_step = 2 / num_tiles_vertical

    # Do the XY plane with Z = 1
    face_key = "Z=1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            # Define corners of the tile
            p1 = Vector(-1 + i * horizontal_step, -1 + j * vertical_step, 1)
            p2 = Vector(-1 + i * horizontal_step, -1 + (j + 1) * vertical_step, 1)
            p3 = Vector(-1 + (i + 1) * horizontal_step, -1 + (j + 1) * vertical_step, 1)
            p4 = Vector(-1 + (i + 1) * horizontal_step, -1 + j * vertical_step, 1)

            # Compute the center of the tile (average of 4 corners)
            center = Vector(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4
            )
            CMP_tile_centers[index_key] = center

    # Do the XY plane with Z = -1
    face_key = "Z=-1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-1 + i * horizontal_step, -1 + j * vertical_step, -1)
            p2 = Vector(-1 + i * horizontal_step, -1 + (j + 1) * vertical_step, -1)
            p3 = Vector(-1 + (i + 1) * horizontal_step, -1 + (j + 1) * vertical_step, -1)
            p4 = Vector(-1 + (i + 1) * horizontal_step, -1 + j * vertical_step, -1)

            center = Vector(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4
            )
            CMP_tile_centers[index_key] = center

    # Do the XZ plane with Y = 1
    face_key = "Y=1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-1 + i * horizontal_step, 1, -1 + j * vertical_step)
            p2 = Vector(-1 + i * horizontal_step, 1, -1 + (j + 1) * vertical_step)
            p3 = Vector(-1 + (i + 1) * horizontal_step, 1, -1 + (j + 1) * vertical_step)
            p4 = Vector(-1 + (i + 1) * horizontal_step, 1, -1 + j * vertical_step)

            center = Vector(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4
            )
            CMP_tile_centers[index_key] = center

    # Do the XZ plane with Y = -1
    face_key = "Y=-1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-1 + i * horizontal_step, -1, -1 + j * vertical_step)
            p2 = Vector(-1 + i * horizontal_step, -1, -1 + (j + 1) * vertical_step)
            p3 = Vector(-1 + (i + 1) * horizontal_step, -1, -1 + (j + 1) * vertical_step)
            p4 = Vector(-1 + (i + 1) * horizontal_step, -1, -1 + j * vertical_step)

            center = Vector(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4
            )
            CMP_tile_centers[index_key] = center

    # Do the YZ plane with X = 1
    face_key = "X=1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(1, -1 + i * horizontal_step, -1 + j * vertical_step)
            p2 = Vector(1, -1 + i * horizontal_step, -1 + (j + 1) * vertical_step)
            p3 = Vector(1, -1 + (i + 1) * horizontal_step, -1 + (j + 1) * vertical_step)
            p4 = Vector(1, -1 + (i + 1) * horizontal_step, -1 + j * vertical_step)

            center = Vector(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4
            )
            CMP_tile_centers[index_key] = center

    # Do the YZ plane with X = -1
    face_key = "X=-1"
    for i in range(num_tiles_horizontal):
        for j in range(num_tiles_vertical):
            index_key = f"face_{face_key}-{i}_{j}"

            p1 = Vector(-1, -1 + i * horizontal_step, -1 + j * vertical_step)
            p2 = Vector(-1, -1 + i * horizontal_step, -1 + (j + 1) * vertical_step)
            p3 = Vector(-1, -1 + (i + 1) * horizontal_step, -1 + (j + 1) * vertical_step)
            p4 = Vector(-1, -1 + (i + 1) * horizontal_step, -1 + j * vertical_step)

            center = Vector(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4
            )
            CMP_tile_centers[index_key] = center

    return CMP_tile_centers



def vector_angle_distance(v1: Vector, v2: Vector) -> float:
    """Computes the angle between two vectors in radians.
    
    Args:
        v1: First vector.
        v2: Second vector.
    
    Returns:
        float: Angle between vectors in radians.
    
    Raises:
        ValidationError: If vectors are invalid.
    """
    try:
        v1_np = np.array([v1.x, v1.y, v1.z])
        v2_np = np.array([v2.x, v2.y, v2.z])
        
        v1_normalized = v1_np / np.linalg.norm(v1_np)
        v2_normalized = v2_np / np.linalg.norm(v2_np)
        
        dot_product = np.dot(v1_normalized, v2_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        return np.arccos(dot_product)
        
    except Exception as e:
        raise ValidationError(f"Error calculating vector angle: {str(e)}")


def find_geodesic_distances(
    vector: Vector,
    vectors: List[Vector]
) -> np.ndarray:
    """Finds geodesic distances between a single Vector and a list of Vectors.
    
    Args:
        vector: Reference vector.
        vectors: List of Vectors.
    
    Returns:
        np.ndarray: Array of [tile_index, geodesic_distance] pairs.
    """
    distances = np.array([
        [int(i), vector_angle_distance(vector, center)]
        for i, center in enumerate(vectors)
    ])
    return distances

def find_geodesic_distances_from_dict(
    vector: Vector,
    tile_centers: Dict[str, Vector]
) -> np.ndarray:
    """
    Finds geodesic distances between a vector and tile centers from a dictionary.

    Args:
        vector: Reference vector.
        tile_centers: Dictionary where keys are tile IDs and values are Vector centers.

    Returns:
        np.ndarray: Array of [tile_key, geodesic_distance] pairs as a structured array.
    """
    distances = np.array([
        (key, vector_angle_distance(vector, center))
        for key, center in tile_centers.items()
    ], dtype=[("tile_key", "U50"), ("geodesic_distance", "f8")])
    
    return distances

def find_ERP_distance(v1: Vector, v2: Vector) -> float:
    """Computes the distance in latitude and longitude (ERP), in radians, between two vectors.
    
    Args:
        v1: First vector.
        v2: Second vector.
    
    Returns:
        float: Distance in ERP between vectors.
    
    Raises:
        ValidationError: If vectors are invalid.
    """
    try:
        radial1 = v1.to_spherical()
        radial2 = v2.to_spherical()

        lat_dist = np.radians(radial1.lat) - np.radians(radial2.lat)
        lon_dist = np.radians(radial1.lon) - np.radians(radial2.lon)
        
        ERP_dist = np.sqrt(lat_dist ** 2 + lon_dist ** 2)

        return ERP_dist
        
    except Exception as e:
        raise ValidationError(f"Error calculating ERP distance: {str(e)}")

def find_ERP_distances_from_dict(
  vector: Vector,
    tile_centers: Dict[str, Vector]
    ) -> np.ndarray:
    """
    Finds ERP distances between a vector and tile centers from a dictionary.

    Args:
        vector: Reference vector.
        tile_centers: Dictionary where keys are tile IDs and values are Vector centers.

    Returns:
        np.ndarray: Array of [tile_key, ERP_distance] pairs as a structured array.
    """
    distances = np.array([
        (key, find_ERP_distance(vector, center))
        for key, center in tile_centers.items()
    ], dtype=[("tile_key", "U50"), ("ERP_distance", "f8")])

    return distances


def validate_video_dimensions(width: int, height: int) -> None:
    """Validates video dimensions.
    
    Args:
        width (int): Video width in pixels.
        height (int): Video height in pixels.
    
    Raises:
        ValidationError: If dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValidationError("Video dimensions must be positive")
    if width % 2 != 0 or height % 2 != 0:
        raise ValidationError("Video dimensions must be even numbers")


def normalize_to_pixel(normalized: np.ndarray, dimension: int) -> np.ndarray:
    """Converts normalized coordinates to pixel coordinates.
    
    Args:
        normalized (np.ndarray): Normalized coordinates (range [0, 1]).
        dimension (int): Dimension (width or height) for scaling.
    
    Returns:
        np.ndarray: Pixel coordinates.
    
    Raises:
        ValidationError: If normalized values are outside [0, 1] range.
    """
    if np.any((normalized < 0) | (normalized > 1)):
        raise ValidationError("Normalized coordinates must be between 0 and 1")
    if dimension <= 0:
        raise ValidationError("Dimension must be positive")
    
    return (normalized * dimension).astype(int)


def pixel_to_spherical(point: Pixel, video_width: int, video_height: int) -> RadialPoint:
    """Converts pixel coordinates to spherical coordinates.
    
    Args:
        point (Pixel): Pixel in pixel coordinates.
        video_width (int): Width of the video in pixels.
        video_height (int): Height of the video in pixels.
    
    Returns:
        RadialPoint: Point in spherical coordinates.
    
    Raises:
        ValidationError: If coordinates are invalid.
    """
    validate_video_dimensions(video_width, video_height)
    
    if point.pixel_x > video_width or point.pixel_y > video_height:
        raise ValidationError("Pixel coordinates exceed video dimensions")
    
    lon = (point.pixel_x / video_width) * 360 - 180
    lat = 90 - (point.pixel_y / video_height) * 180
    
    return RadialPoint(lon=lon, lat=lat)


def process_viewport_data(
    filepath: Union[str, Path],
    video_width: int,
    video_height: int
) -> Tuple[pd.DataFrame, str]:
    """Processes viewport center trajectory data from a CSV file.
    
    Args:
        filepath (Union[str, Path]): Path to the CSV file.
        video_width (int): Width of the video in pixels.
        video_height (int): Height of the video in pixels.
    
    Returns:
        Tuple[pd.DataFrame, str]: Processed data and trajectory identifier.
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValidationError: If data format is invalid.
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read and validate data
        data = pd.read_csv(filepath, usecols=["time", "2dmu", "2dmv"]).dropna()
        if data.empty:
            raise ValidationError(f"No valid data found in {filepath}")
        
        # Normalize time to start at 0
        data["time"] -= data["time"].min()
        
        # Convert normalized coordinates to pixel coordinates
        data["pixel_x"] = normalize_to_pixel(data["2dmu"].values, video_width)
        data["pixel_y"] = normalize_to_pixel(data["2dmv"].values, video_height)
        
        # Convert to spherical coordinates
        spherical_coords = data.apply(
            lambda row: pixel_to_spherical(
                Pixel(row["pixel_x"], row["pixel_y"]),
                video_width,
                video_height
            ),
            axis=1
        )
        data["lon"], data["lat"] = zip(*[(p.lon, p.lat) for p in spherical_coords])
        
        # Generate identifier from filename
        identifier = filepath.stem
        
        return data, identifier
    
    except Exception as e:
        raise ValidationError(f"Error processing viewport data: {str(e)}")


def format_trajectory_data(
    trajectory_data: List[Tuple[str, pd.DataFrame]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Formats trajectory data for analysis.
    
    Args:
        trajectory_data: List of tuples containing (identifier, data) pairs.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Radial Points and Vectors at each time step.
    
    Raises:
        ValidationError: If data format is invalid.
    """
    if not trajectory_data:
        raise ValidationError("No trajectory data provided")
    
    # Initialize column names
    columns = ["time"] + [pair[0] for pair in trajectory_data]
    
    # Initialize dictionaries for points and vectors
    points_dict = {col: [] for col in columns}
    vectors_dict = {col: [] for col in columns}
    
    # Process each trajectory
    for identifier, data in trajectory_data:
        data["time"] = data["time"].round(1)  # Round time to nearest tenth
        
        for row in data.itertuples(index=False):
            time = row.time
            
            # Add time point if not exists
            if time not in points_dict["time"]:
                points_dict["time"].append(time)
                vectors_dict["time"].append(time)
                
                # Initialize None for all trajectories at this time
                for col in columns[1:]:
                    points_dict[col].append(None)
                    vectors_dict[col].append(None)
            
            # Get index for this time
            idx = points_dict["time"].index(time)
            
            # Create and store RadialPoint
            lon = round(row.lon, 1)
            lat = round(row.lat, 1)
            
            # Normalize coordinates if needed
            if lon <= -180:
                lon = (lon + 360) % 360 - 180
            if lat <= -90:
                lat = (lat + 180) % 180 - 90
            
            radial_point = RadialPoint(lon=lon, lat=lat)
            points_dict[identifier][idx] = radial_point
            
            # Create and store Vector
            vector = Vector.from_spherical(lon, lat)
            vectors_dict[identifier][idx] = vector
    
    # Create DataFrames
    points_df = pd.DataFrame(points_dict)
    vectors_df = pd.DataFrame(vectors_dict)
    
    return points_df, vectors_df

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector.

    Args:
        v: A np.ndarray for a vector.

    Returns:
        np.ndarray: The normalized vector.
    """
    return v / np.linalg.norm(v)

def find_perpendicular_on_tangent_plane(vec: np.ndarray, midpoint: np.ndarray) -> np.ndarray:
    """
    Find a vector perpendicular to `vec` that lies on the tangent plane at the midpoint on the sphere.

    Args:
        vec: Vector between two points on the sphere.
        midpoint: Midpoint between the two tile centers (point on the sphere).
    Returns:
        np.ndarray: A vector perpendicular to `vec` on the tangent plane at the midpoint.
    """
    # Normalize the midpoint to get the radial vector
    radial_vec = normalize(midpoint)

    # Compute the cross product between the radial vector and the vector `vec`
    perp_vec = np.cross(radial_vec, vec)

    # Normalize the perpendicular vector to ensure it lies on the tangent plane
    perp_vec = normalize(perp_vec)

    return perp_vec

def great_circle_intersection(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """
    Calculate the intersection points of two great circles on a unit sphere.

    Args:
        n1: Normal vector of the first great circle.
        n2: Normal vector of the second great circle.
    Returns:
        Two intersection points on the unit sphere (each is a 3D vector).
    """
    # Normalize the normal vectors (ensures they are unit vectors)
    n1 = normalize(n1)
    n2 = normalize(n2)

    # Find the direction of the line of intersection (cross product of the two normal vectors)
    line_direction = np.cross(n1, n2)
    line_direction = normalize(line_direction)  # Normalize the direction vector

    # The two intersection points are the normalized line direction and its inverse
    p1 = line_direction
    p2 = -line_direction

    return p1, p2

def get_line_segment(v1: Vector, v2: Vector) -> np.ndarray:
    """
    Calculate the line segment between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.
    Returns:
        np.ndarray: the line segment between the vectors.
    """
    line_segment = np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z])

    return line_segment

def find_nearest_point(v1: Vector, v2: Vector, compare_vector: Vector) -> Vector:
    """
    Calculate the nearest point to the compared point.

    Args:
        p1: First vector.
        p2: Second vector.
        compare_vector: The point to compare.
    Returns:
        The nearest vector.
    """
    v1_i_seg = get_line_segment(compare_vector, v1)
    v2_i_seg = get_line_segment(compare_vector, v2)
    length_i_v1 = np.linalg.norm(v1_i_seg).round(4)
    length_i_v2 = np.linalg.norm(v2_i_seg).round(4)
    
    if (length_i_v1 < length_i_v2):
        return v1
    else:
        return v2

def spherical_interpolation(v1: Vector, v2: Vector, t: float) -> np.ndarray:
    """Perform spherical linear interpolation (slerp) between two points on the sphere (takes the shorter path).
    
    Args:
        v1: The Vector representing the first point on the sphere.
        v2: The Vector representing the second point on the sphere.

    Returns:
        np.ndarray: An array for the vector of the spherical linear interpolation.
    """
    p1 = np.array([v1.x, v1.y, v1.z])
    p2 = np.array([v2.x, v2.y, v2.z])

    # Normalize the vectors
    p1 = normalize(p1)
    p2 = normalize(p2)

    dot_product = np.dot(p1, p2)
    # Clip dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle between vectors
    theta = np.arccos(dot_product)

    # Compute slerp
    return (np.sin((1 - t) * theta) * p1 + np.sin(t * theta) * p2) / np.sin(theta)

def get_tile_corners(tile_boundaries: List[List[Vector]]) -> List[Vector]:
    """Get tile corners that follows edge order given a list of tile boundaries. This means that each consecutive pair is an edge on the tile.
    
    Args:
        tile_boundaries: A list of a tile boundaries. Each tile boundary is a list of two points with the vector representing the point on the sphere.

    Returns:
        list[Vector]: A list of tile corners that follows edge order.
    """

    decimals_to_round = 4

    tile_corners = [tile_boundaries[0][0].round(decimals=decimals_to_round), tile_boundaries[0][1].round(decimals=decimals_to_round)]
    tile_corners_dict = {}
    tile_corners_dict[tile_corners[0]] = True
    tile_corners_dict[tile_corners[1]] = True

    edge_dict = {}

    for P1, P2 in tile_boundaries:
        P1_rounded = P1.round(decimals=decimals_to_round)
        P2_rounded = P2.round(decimals=decimals_to_round)

        if P1_rounded not in edge_dict:
            edge_dict[P1_rounded] = [P2_rounded]
        else:
            edge_dict[P1_rounded].append(P2_rounded)
        
        if P2_rounded not in edge_dict:
            edge_dict[P2_rounded] = [P1_rounded]
        else:
            edge_dict[P2_rounded].append(P1_rounded)

        if P1_rounded not in tile_corners_dict:
            tile_corners_dict[P1_rounded] = False
        if P2_rounded not in tile_corners_dict:
            tile_corners_dict[P2_rounded] = False
    
    # Check and make sure that each corner has exactly two adjacent corners. If not, remove then and notify the user.
    for corner in list(edge_dict.keys()):
        # If a corner has less than 2 edges, then remove it and all other references to it.
        if (corner not in edge_dict):
          continue
        elif(len(edge_dict[corner]) == 0):
          del edge_dict[corner]
        elif len(edge_dict[corner]) < 2:
            neighbor = edge_dict[corner][0]

            print(f"Corner {corner} was found with only one neighbor, and is being removed (due to it being an extraneous edge).")
            print(f"This extraneous edge is: ({corner}, {neighbor}), tile: {edge_dict}")
            print("Fibonacci lattice points have only been tested for up to 3200 points." +
        "This is because the method to generate tile boundaries (the actual edges) for area calculation is not robust in the interest of decreasing compute time." +
        "See 'furthest_search_factor' under get_FB_tile_boundaries for more info on how nearby neighbors for candidate tile centers are filtered." +
        "Assigning points to tiles is robust, this is merely for area calculation and visualization purposes.")

            edge_dict[neighbor].remove(corner)
            del edge_dict[corner]
            del tile_corners_dict[corner]
        # If a corner has an edge with itself, remove that edge.
        elif len(edge_dict[corner]) > 2:
            index = 0
            while index < len(edge_dict[corner]):
                other_corner = edge_dict[corner][index]
                if other_corner == corner:
                    edge_dict[corner].remove(other_corner)
                else:
                    index += 1
            

    next_edge = tile_corners[1]

    try:
        while (tile_corners_dict[edge_dict[next_edge][0]] == False or tile_corners_dict[edge_dict[next_edge][1]] == False):
            next_edge = edge_dict[next_edge][0] if tile_corners_dict[edge_dict[next_edge][0]] == False else edge_dict[next_edge][1]
            tile_corners.append(next_edge)
            tile_corners_dict[next_edge] = True
    except Exception as e:
        print(f"Exception made on next edge: {next_edge}, tile: {tile_boundaries}")
        print(f"Edge dict: {edge_dict}")
        print(e)
        print("Fibonacci lattice points have only been tested for up to 3200 points." +
        "This is because the method to generate tile boundaries (the actual edges) for area calculation is not robust in the interest of decreasing compute time." +
        "See 'furthest_search_factor' under get_FB_tile_boundaries for more info on how nearby neighbors for candidate tile centers are filtered." +
        "Assigning points to tiles is robust, this is merely for area calculation and visualization purposes.")

    return tile_corners

def triangulate_spherical_polygon(tile_corners: List[Vector]) -> List[List[Vector]]:
    """
    Triangulate a spherical polygon by using a fan method.
    - The first point is used as the fixed anchor.
    - The rest of the polygon is divided into triangles from that anchor.

    Args:
        tile_corners: A list of points on the sphere that represent the corners of the spherical tile.

    Returns:
        listlist[[Vector]]: A list of triangles that represent the triangulation of the spherical tile.

    """
    if len(tile_corners) < 3:
        raise ValueError("At least 3 boundary points are needed for a polygon.")

    triangles = []
    anchor = tile_corners[0]  # Fixed point for fan triangulation

    for i in range(1, len(tile_corners) - 1):
        triangles.append([anchor, tile_corners[i], tile_corners[i + 1]])

    return triangles

def angle_at_vertex(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """Compute the angle at v1 formed by the great circles (v1, v2) and (v1, v3).
    
    Args:
      v1: A vector representing the point v1 on the sphere.
      v2: A vector representing the point v2 on the sphere.
      v3: A vector representing the point v3 on the sphere.

    Returns:
      float: The angle at v1 formed by the great circles (v1, v2) and (v1, v3).
    """
    
    t1 = v2 - np.dot(v2, v1) * v1
    t2 = v3 - np.dot(v3, v1) * v1

    t1 /= np.linalg.norm(t1)
    t2 /= np.linalg.norm(t2)

    # Compute the angle
    return np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0))  # Clip to avoid precision errors

def calculate_spherical_triangle_area(P1: Vector, P2: Vector, P3: Vector, radius: float = 1.0) -> float:
    """
    Calculate the surface area of a spherical triangle given the vectors of points of the spherical triangle.
    of the great circles defining the triangle.

    Args:
      P1: A vector representing the point P1 on the sphere.
      P2: A vector representing the point P2 on the sphere.
      P3: A vector representing the point P3 on the sphere.
      radius: The radius of the sphere.

    Returns:
      float: The surface area of the spherical triangle.
    """

    P1_np = np.array([P1.x, P1.y, P1.z])
    P2_np = np.array([P2.x, P2.y, P2.z])
    P3_np = np.array([P3.x, P3.y, P3.z])

    # Ensure the points are unit vectors
    V1 = P1_np / np.linalg.norm(P1_np)
    V2 = P2_np / np.linalg.norm(P2_np)
    V3 = P3_np / np.linalg.norm(P3_np)

    # Compute the three interior angles at P1, P2, and P3
    A = angle_at_vertex(V1, V2, V3)
    B = angle_at_vertex(V2, V3, V1)
    C = angle_at_vertex(V3, V1, V2)

    # Spherical excess formula
    E = A + B + C - np.pi

    # Surface area of the spherical triangle
    return E * (radius ** 2)

def compute_spherical_polygon_area(tile_boundaries: List[List[Vector]], radius=1.0) -> float:
    """
    Compute the total area of a convex spherical polygon by summing the areas of its triangulated parts.

    Args:
      tile_boundaries: A list of a tile boundaries. Each tile boundary is a list of two points with the vector representing the point on the sphere.
      radius: The radius of the sphere.

    Returns:
      float: The total area of the spherical polygon.
    """

    tile_corners = get_tile_corners(tile_boundaries)
    triangles = triangulate_spherical_polygon(tile_corners)

    total_area = 0.0

    for triangle in triangles:
      triangle_area = calculate_spherical_triangle_area(triangle[0], triangle[1], triangle[2], radius)
      total_area += triangle_area

    return total_area

def compute_FB_tile_areas(tile_count: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute the fraction of the sphere each tile occupies and the tile areas for Fibonacci lattice tiling.

    Args:
        tile_count (int): Number of tiles.

    Returns:
        tuple: (fraction_of_sphere_dict, tile_area_dict)

    Raises:
        ValueError: If tile_count is not a positive integer.
    """

    if tile_count <= 0:
        raise ValidationError("Number of points must be positive!")

    furthest_search_factor = 1.7

    tile_boundaries_dict = get_FB_tile_boundaries(tile_count, furthest_search_factor)

    fraction_of_sphere_dict = {}
    tile_area_dict = {}
    total_sphere_area = 4 * np.pi

    for tile_boundaries_index, tile_boundaries in tile_boundaries_dict.items():
        tile_area = compute_spherical_polygon_area(tile_boundaries)
        tile_area_dict[tile_boundaries_index] = tile_area

        fraction_of_sphere = tile_area / total_sphere_area
        fraction_of_sphere_dict[tile_boundaries_index] = fraction_of_sphere

    return tile_area_dict, fraction_of_sphere_dict

def compute_ERP_tile_areas(num_tiles_horizontal: int, num_tiles_vertical: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute the fraction of the sphere each tile occupies and the tile areas for ERP tiling.

    Args:
        tile_count (int): Number of tiles.

    Returns:
        tuple: (fraction_of_sphere_dict, tile_area_dict)

    Raises:
        ValueError: If either num_tiles_horizontal or num_tiles_vertical is not a positive integer.
    """

    if num_tiles_horizontal <= 0 or num_tiles_vertical <= 0:
        raise ValidationError("Number of tiles horizontal and vertical must be positive!")

    tile_boundaries_dict = get_ERP_tile_boundaries(num_tiles_horizontal, num_tiles_vertical)

    fraction_of_sphere_dict = {}
    tile_area_dict = {}
    total_sphere_area = 4 * np.pi

    for tile_boundaries_index, tile_boundaries in tile_boundaries_dict.items():
        tile_area = compute_spherical_polygon_area(tile_boundaries)
        tile_area_dict[tile_boundaries_index] = tile_area

        fraction_of_sphere = tile_area / total_sphere_area
        fraction_of_sphere_dict[tile_boundaries_index] = fraction_of_sphere

    return tile_area_dict, fraction_of_sphere_dict

def compute_CMP_tile_areas(num_tiles_horizontal: int, num_tiles_vertical: int, radius: float = 1.0) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute the fraction of the sphere each tile occupies and the tile areas for CMP tiling.
    Total number of tiles will be 6 * num_tiles_horizontal * num_tiles_vertical.

    Args:
        tile_count (int): Number of tiles.
        num_tiles_horizontal: The number of horizontal spatial bins to tile a face of the CMP cube with.
        num_tiles_vertical: The number of vertical spatial bins to tile a face of the CMP cube with.
        radius: the radius of the circle.

    Returns:
        tuple: (fraction_of_sphere_dict, tile_area_dict)

    Raises:
        ValueError: If either num_tiles_horizontal or num_tiles_vertical is not a positive integer.
    """
    if num_tiles_horizontal <= 0 or num_tiles_vertical <= 0:
        raise ValidationError("Number of tiles horizontal and vertical must be positive!")

    tile_boundaries_dict = get_CMP_tile_boundaries(num_tiles_horizontal, num_tiles_vertical, radius)

    fraction_of_sphere_dict = {}
    tile_area_dict = {}
    total_sphere_area = 4 * radius * radius * np.pi

    for tile_boundaries_index, tile_boundaries in tile_boundaries_dict.items():
        tile_area = compute_spherical_polygon_area(tile_boundaries)
        tile_area_dict[tile_boundaries_index] = tile_area

        fraction_of_sphere = tile_area / total_sphere_area
        fraction_of_sphere_dict[tile_boundaries_index] = fraction_of_sphere

    return tile_area_dict, fraction_of_sphere_dict