"""Heatmap utilities module for spherical tiling and heatmap visualizations.

This module provides utilities for managing heatmap configurations and calculating heatmap weights.
It includes functions for calculating heat functions, and generating heat across a set of tiles for a single Vector.

Functions:
    calculate_tile_weights: Computes weight distribution across tiles, returning dictionary of weight by tile Vector as key.
    calculate_tile_weights_by_index: Computes weight distribution across tiles, returning dictionary of weight by tile index as key.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
import math

from fibonacci_lattice_tiling_toolkit import Vector, ValidationError
from fibonacci_lattice_tiling_toolkit.utilities.data_utils import find_geodesic_distances, find_geodesic_distances_from_dict, find_ERP_distances_from_dict


class HeatFunctionType(Enum):
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    NEAREST_NEIGHBOR = "nearest_neighbor"

class HeatFunction:
    def __init__(self, heat_function_type):
        self.heat_function_type = heat_function_type

    def __call__(self, x: float) -> float:
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"Input to heat function must be in [0, 1]. Got: {x}")
        return self._compute(x)

    def _compute(self, x: float) -> float:
        """To be implemented by subclasses."""
        raise NotImplementedError


class GaussianHeatFunction(HeatFunction):
    def __init__(self, sigma: float = 0.3):
        super().__init__(HeatFunctionType.GAUSSIAN)
        self.sigma = sigma

    def _compute(self, x: float) -> float:
        # Standard Gaussian centered at 0
        return math.exp(-(x ** 2) / (2 * self.sigma ** 2))


class ExponentialHeatFunction(HeatFunction):
    def __init__(self, decay_factor: float = 2.0):
        super().__init__(HeatFunctionType.EXPONENTIAL)
        self.decay_factor = decay_factor
    
    def _compute(self, x: float) -> float:
        # decay from 0 to 1
        return math.exp(-self.decay_factor * x)


class UniformHeatFunction(HeatFunction):
    def __init__(self):
        super().__init__(HeatFunctionType.UNIFORM)
    
    def _compute(self, x: float) -> float:
        return 1.0


class NearestNeighborHeatFunction(HeatFunction):
    def __init__(self):
        super().__init__(HeatFunctionType.NEAREST_NEIGHBOR)

@dataclass
class HeatmapConfig:
    """Configuration for heatmap calculations.
    
    Attributes:
        fov_angle (float): Field of view angle in degrees.
        use_erroneous_ERP_distance (bool): True if using the ERP Euclidean distance. If false, use geodesic distance on sphere.
        heat_function (HeatFunction): The specified heat function to use. It takes in a value from 0 to 1 and returns the given heat.
    """
    fov_angle: float = 120.0
    use_erroneous_ERP_distance: bool = False
    
    heat_function: HeatFunction = field(default_factory=GaussianHeatFunction)

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if not 0 < self.fov_angle <= 360:
            raise ValidationError("FOV angle must be between 0 and 360 degrees")

def find_nearest_tile(
        vector: Vector,
        tile_centers: List[Vector]
)-> int:
    """
    Find nearest tile for a given vector.

    Args:
        vector: Input vector.
        tile_centers: List of tile center vectors.
    
    Returns:
        int: the index of tile_centers for the closest tile.
    """
    distances = find_geodesic_distances(vector, tile_centers)
    nearest_tile = int(distances[np.argmin(distances[:, 1])][0])

    return nearest_tile

def calculate_tile_weights(
    vector: Vector,
    tile_centers: List[Vector],
    config: HeatmapConfig
) -> Dict[Vector, float]:
    """Calculates weight distribution across tiles for a vector using fibonacci lattice tiling.
    
    Args:
        vector: Input vector.
        tile_centers: List of tile center vectors.
        config: Heatmap calculation configuration.
    
    Returns:
        Dict[Vector, float]: Dictionary mapping tile centers to weights.
    """
    weights = {}
    max_distance = np.radians(config.fov_angle / 2.0)
    
    # Calculate geodesic or ERP distances based on config
    if not config.use_erroneous_ERP_distance:
      distances = find_geodesic_distances_from_dict(vector, tile_centers)
    else:
      distances = find_ERP_distances_from_dict(vector, tile_centers)

    distances = sorted(distances, key=lambda x: x[1])
    
    if config.heat_function.heat_function_type != HeatFunctionType.NEAREST_NEIGHBOR:
        # Distribute weights based on geodesic distance
        for tile_idx, distance in distances:
            if distance < max_distance:
                tile = tile_centers[int(tile_idx)]
                normalized_distance = distance / max_distance
                weight = config.heat_function(normalized_distance)
                weights[tile] = weight
            else:
                break
    else:
        # Assign weight only to nearest tile
        nearest_idx = int(distances[0][0])
        weights[tile_centers[nearest_idx]] = 1.0
    
    return weights

def calculate_tile_weights_by_index(
    vector: Vector,
    tile_centers: Dict[str, Vector],
    config: HeatmapConfig
) -> Dict[str, float]:
    """Calculates weight distribution across tiles for a vector using a set of tile centers.

    Args:
        vector: Input vector.
        tile_centers: List of tile center vectors.
        config: Heatmap calculation configuration.

    Returns:
        Dict[str, float]: Dictionary mapping tile index to weights.
    """
    weights = {}
    max_distance = np.radians(config.fov_angle / 2.0)

    # Calculate geodesic or ERP distances based on config
    if not config.use_erroneous_ERP_distance:
      distances = find_geodesic_distances_from_dict(vector, tile_centers)
    else:
      distances = find_ERP_distances_from_dict(vector, tile_centers)
    
    distances = sorted(distances, key=lambda x: x[1])

    if config.heat_function.heat_function_type != HeatFunctionType.NEAREST_NEIGHBOR:
        # Distribute weights based on geodesic distance
        for tile_idx, distance in distances:
            tile_idx_str = str(tile_idx)
            if distance < max_distance:
                normalized_distance = distance / max_distance
                weight = config.heat_function(normalized_distance)
                weights[tile_idx_str] = weight
            else:
                break
    else:
        # Assign weight to only to the nearest tile
        nearest_tile_idx_str = distances[0][0]
        weights[nearest_tile_idx_str] = 1.0

    return weights

def compute_heatmap(
    vectors: List[Vector],
    tile_centers: Dict[str, Vector],
    config: HeatmapConfig
) -> Dict[str, float]:
    """Calculates weight distribution across tiles for a set of vectors using a set of tile centers.

    Args:
        vectors: Input list of vectors.
        tile_centers: List of tile center vectors.
        config: Heatmap calculation configuration.

    Returns:
        Dict[str, float]: Dictionary mapping tile index to weights.
    """

    tile_weights = {}

    for tile_center_key in tile_centers.keys():
        tile_weights[tile_center_key] = 0.0

    for vector in vectors:
        tile_weights_list = calculate_tile_weights_by_index(
            vector=vector,
            tile_centers=tile_centers,
            config=config)

        for tile_index, tile_weight in tile_weights_list.items():
            tile_weights[tile_index] += tile_weight
    
    return tile_weights