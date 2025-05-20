"""Tests for spherical tiling and heatmap visualizations."""

from fibonacci_lattice_tiling_toolkit.data_types import Point, Pixel, RadialPoint, Vector
from fibonacci_lattice_tiling_toolkit.utilities.data_utils import get_FB_tile_boundaries, get_ERP_tile_boundaries, get_CMP_tile_boundaries, generate_fibonacci_lattice, get_CMP_tile_centers
from fibonacci_lattice_tiling_toolkit.utilities.heatmap_utils import HeatmapConfig, HeatFunctionType, GaussianHeatFunction, ExponentialHeatFunction, UniformHeatFunction, NearestNeighborHeatFunction, calculate_tile_weights

def test_point_creation():
    """Test Point class creation."""
    point = Point(x=200, y=-100)
    assert point.x == 200
    assert point.y == -100

def test_pixel_creation():
    """Test Pixel class creation."""
    pixel = Pixel(pixel_x=100, pixel_y=200)
    assert pixel.pixel_x == 100
    assert pixel.pixel_y == 200

def test_radial_point_creation():
    """Test RadialPoint class creation."""
    point = RadialPoint(lon=45.0, lat=30.0)
    assert point.lon == 45.0
    assert point.lat == 30.0

def test_vector_creation():
    """Test Vector class creation."""
    vector = Vector(x=1.0, y=2.0, z=3.0)
    assert vector.x == 1.0
    assert vector.y == 2.0
    assert vector.z == 3.0

def test_tiling_generation():
    """Test Tilng Generation functions."""
    get_FB_tile_boundaries(49)
    get_ERP_tile_boundaries(num_tiles_horizontal=10, num_tiles_vertical=5)
    get_CMP_tile_boundaries(num_tiles_horizontal=5, num_tiles_vertical=5)

def test_heatmap_config():
    """Test Heatmap Config creation."""
    gaussian_config = HeatmapConfig(
        use_erroneous_ERP_distance=False,
        heat_function=GaussianHeatFunction()
    )
    assert gaussian_config.heat_function.heat_function_type == HeatFunctionType.GAUSSIAN
    assert gaussian_config.use_erroneous_ERP_distance == False

    exponential_config = HeatmapConfig(
        use_erroneous_ERP_distance=True,
        heat_function=GaussianHeatFunction()
    )

    assert exponential_config.heat_function.heat_function_type == HeatFunctionType.EXPONENTIAL
    assert exponential_config.use_erroneous_ERP_distance == True

    uniform_config = HeatmapConfig(
        heat_function=UniformHeatFunction()
    )

    assert uniform_config.heat_function.heat_function_type == HeatFunctionType.UNIFORM
    assert uniform_config.use_erroneous_ERP_distance == False

    nearest_neighbor_config = HeatmapConfig(
        heat_function=NearestNeighborHeatFunction()
    )

    assert nearest_neighbor_config.heat_function.heat_function_type == HeatFunctionType.NEAREST_NEIGHBOR
    assert nearest_neighbor_config.use_erroneous_ERP_distance == False

def test_weight_generation():
    """Test Tiling Weight Generation functions."""
    vector = Vector(x=1.0, y=2.0, z=3.0)
    
    fb_tile_centers = generate_fibonacci_lattice(49)

    config = HeatmapConfig()

    calculate_tile_weights(
        vector=vector,
        tile_centers=fb_tile_centers,
        config=config
    )


