from fibonacci_lattice_tiling_toolkit.utilities.data_utils import generate_fibonacci_lattice
from fibonacci_lattice_tiling_toolkit.data_types import Vector, ZThetaPoint
from fibonacci_lattice_tiling_toolkit.utilities.visualization_utils import save_tiling_visualization_image, save_tiling_visualization_video
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
import pandas
from pathlib import Path


FB_count = 65
layout_file_name = f"SFLP_{FB_count}_wrapped_layout.csv"
output_file_name = f"FB_Projection-({FB_count})_points-wrapped-cols_only.png"

wrap = True
lift_bottom = False
columns = True
rows = False
graph_title = "SFLP on Z-Theta Projection"
xlim = (-320, 180)
ylim = (-1.2, 1.2)
figsize=(20, 12)

def get_col_index(
    first_SFLP_index_in_col: int,         # (N,)
    reference_element_SFLP_index: int,    # (N,)
    first_index_hop: int,
    second_index_hop: int,
    col_generator_vector: Tuple[float, float]                   # tuple (x, y)
):
    """
    GPU-parallelized get_col_index for multiple inputs,
    with col_generator_vector as tuple (x, y).
    """
    multiplier = reference_element_SFLP_index - first_SFLP_index_in_col
    if col_generator_vector[0] > 0:
        multiplier = -multiplier

    if first_index_hop > second_index_hop:
        prev_fb_number = first_index_hop - second_index_hop
        larger_fb_number = first_index_hop
    else:
        prev_fb_number = second_index_hop - first_index_hop
        larger_fb_number = second_index_hop


    mod_larger = multiplier % larger_fb_number

    col_index = ((prev_fb_number % larger_fb_number) * mod_larger) % larger_fb_number

    return col_index

def get_connections(layout_csv_path, columns, rows, wrap: bool = False, lift_bottom: bool = False) -> Dict[str, List[Tuple[int, int]]]:

    connections: Dict[str, List[Tuple[int, int]]] = {}
    connections["green"] = []
    connections["red"] = []

    layout_pd = pandas.read_csv(layout_csv_path, header=None)

    if columns:
        for i in range(len(layout_pd.iloc[0])):
            for j in range(0, len(layout_pd.iloc[:,i]) - 1):
                val_above = layout_pd.iloc[j, i]
                val_below = layout_pd.iloc[j + 1, i]

                if (wrap and lift_bottom) and (val_above < 0 or val_below < 0):
                    connections["red"].append((val_below, val_above))
                elif val_above != val_below and ((wrap) or val_below < val_above) and not (val_above < 0 or val_below < 0):
                    connections["red"].append((val_below, val_above))

    if rows:
        for i in range(len(layout_pd.iloc[0]) - 1):
            for j in range(len(layout_pd.iloc[:,i])):
                val_left = layout_pd.iloc[j, i]
                val_right = layout_pd.iloc[j, i + 1]

                if (wrap and lift_bottom) and (val_left < 0 or val_right < 0):
                    connections["green"].append((val_left, val_right))
                elif val_left != val_right and (wrap or val_left < val_right) and not (val_left < 0 or val_right < 0):
                    connections["green"].append((val_left, val_right))

    return connections

def reorganize_points(points: list[ZThetaPoint], layout_csv_path, wrap: bool, lift_bottom: bool) -> list[ZThetaPoint]:
    reorganized_points = [point for point in points]
    layout_pd = pandas.read_csv(layout_csv_path, header=None)

    col_length = len(layout_pd.iloc[:,0])

    reference_SFLP_index = layout_pd.iloc[col_length - 1, 0]
    pixel_above_reference_index = layout_pd.iloc[col_length - 2, 0]

    first_index_hop = pixel_above_reference_index - reference_SFLP_index
    second_index_hop = layout_pd.iloc[col_length - 1, 1] - reference_SFLP_index

    c_theta = points[pixel_above_reference_index].theta - points[reference_SFLP_index].theta
    c_z = points[pixel_above_reference_index].z - points[reference_SFLP_index].z
    column_generator_vector = (c_theta, c_z)

    for i in range(len(points)):
        col_index = get_col_index(
            first_SFLP_index_in_col=i,
            reference_element_SFLP_index=reference_SFLP_index,
            first_index_hop=first_index_hop,
            second_index_hop=second_index_hop,
            col_generator_vector=column_generator_vector,
        )

        bottom_index = col_index * second_index_hop + reference_SFLP_index
        left_index = i - col_index * second_index_hop

        needs_lift = bottom_index > i

        if left_index < 0:
            left_index = reference_SFLP_index

        needs_rightward_shift = points[i].theta < points[left_index].theta

        if ((bottom_index == 15 and i >= 15)
            or (bottom_index == 20 and i >= 36)
            or (bottom_index == 25 and i >= 57)):
            needs_rightward_shift = False
        else:
            needs_rightward_shift = True

        print(left_index, i, needs_rightward_shift)

        final_point = ZThetaPoint(theta=points[i].theta, z=points[i].z)

        if (not needs_rightward_shift) and wrap:
            final_point.theta -= 360

        if needs_lift and lift_bottom:
            final_point.theta += col_length * column_generator_vector[0]
            final_point.z += col_length * column_generator_vector[1]
        
        reorganized_points[i] = final_point

    
    padded_points = {}

    if lift_bottom:
        for i in range(len(layout_pd.iloc[0])):
            for j in range(len(layout_pd.iloc[:,i])):
                element = layout_pd.iloc[j, i]
                if element < 0:
                    padded_element = int(-1 * element)

                    final_point = ZThetaPoint(theta=reorganized_points[padded_element].theta, z=reorganized_points[padded_element].z, validate=False)

                    final_point.theta += column_generator_vector[0]
                    final_point.z += column_generator_vector[1]

                    padded_points[padded_element] = final_point

    print("Reorganizing points!")

    return reorganized_points, padded_points


def plot_z_theta_projection(
    points: List[ZThetaPoint],
    padded_points: Dict[int, ZThetaPoint],
    title: str, 
    save_path: Path,
    connections: Dict[str, List[Tuple[int, int]]] = None,
    circle_size: int = 2800,
    font_size: int = 32,
    line_width: int = 6,
    line_alpha: float = 0.7,
    xlim: tuple[float, float] = (-200, 200),
    ylim: tuple[float, float] = (-1, 1),
    figsize: tuple[float, float] = (16, 10),
):
    """
    Plots ZThetaPoints with large blue circles, white index labels inside,
    and colored connection lines behind them.
    """
    plt.figure(figsize=figsize)

    xbox = (-180, 180)
    ybox = (-1, 1)

    plt.plot([xbox[0], xbox[1], xbox[1], xbox[0], xbox[0]], [ybox[0], ybox[0], ybox[1], ybox[1], ybox[0]], 
        color="black", linewidth=2, alpha=0.5, zorder=1, clip_on=True)
    
    # Draw connections (Lines) - zorder=1
    if connections:
        for color, index_pairs in connections.items():
            for idx1, idx2 in index_pairs:
                if idx1 < len(points) and idx1 >= 0 and idx2 < len(points) and idx2 >= 0:
                    p1, p2 = points[idx1], points[idx2]
                    if abs(p1.theta - p2.theta) < 180:
                        plt.plot([p1.theta, p2.theta], [p1.z, p2.z], 
                                color=color, linewidth=line_width, alpha=line_alpha, zorder=1, clip_on=True)
                    else:
                        if p1.theta < p2.theta:
                            p1, p2 = p2, p1
                        
                        p2_wrapped_theta = p2.theta + 360
                        p1_wrapped_theta = p1.theta - 360

                        # Compute the wrap around points and where the line hits the edge of the box.

                        m_to_right = (p2.z - p1.z) / (p2_wrapped_theta - p1.theta)
                        z_at_180 = p1.z + m_to_right * (180 - p1.theta)
                        
                        plt.plot([p1.theta, 180], [p1.z, z_at_180], 
                                color=color, linewidth=line_width, alpha=line_alpha, linestyle='--', zorder=1, clip_on=True)

                        m_to_left = (p1.z - p2.z) / (p1_wrapped_theta - p2.theta)
                        z_at_neg_180 = p2.z + m_to_left * (-180 - p2.theta)

                        plt.plot([p2.theta, -180], [p2.z, z_at_neg_180], 
                                color=color, linewidth=line_width, alpha=line_alpha, linestyle='--', zorder=1, clip_on=True)
                
                elif idx2 < 0 and idx1 < len(points) and idx1 > 0:
                    p1, p2 = points[idx1], padded_points[int(-1 * idx2)]
                    plt.plot([p1.theta, p2.theta], [p1.z, p2.z], 
                                color=color, linewidth=line_width, alpha=line_alpha, zorder=1, clip_on=True)
                elif idx1 < 0 and idx2 < len(points) and idx2 > 0:
                    p1, p2 = padded_points[int(-1 * idx1)], points[idx2]
                    plt.plot([p1.theta, p2.theta], [p1.z, p2.z], 
                                color=color, linewidth=line_width, alpha=line_alpha, zorder=1, clip_on=True)
                elif idx1 < 0 and idx2 < 0:
                    p1, p2 = padded_points[int(-1 * idx1)], padded_points[int(-1 * idx2)]
                    plt.plot([p1.theta, p2.theta], [p1.z, p2.z], 
                                color=color, linewidth=line_width, alpha=line_alpha, zorder=1, clip_on=True)



    # Draw the points (Large Circles) - zorder=2
    thetas = [p.theta for p in points]
    zs = [p.z for p in points]
    plt.scatter(thetas, zs, color='blue', s=circle_size, zorder=2, 
                edgecolors='black', linewidth=0.5 * line_width, clip_on=True)
    

    # Add indices in white text - zorder=3
    for i, p in enumerate(points):
        plt.text(
            p.theta, p.z, str(i), 
            color='white', 
            fontsize=font_size, 
            ha='center',        # Horizontal alignment
            va='center',        # Vertical alignment
            fontweight='bold',  # Makes it easier to read on blue
            zorder=3,
            clip_on = True
        )

    # Draw the padded points (Large Circles) - zorder=2
    if padded_points:
        thetas = [p.theta for p in padded_points.values()]
        zs = [p.z for p in padded_points.values()]
        plt.scatter(thetas, zs, color='black', s=circle_size, zorder=2, 
                    edgecolors='black', linewidth=0.5 * line_width, clip_on=True)
    

        # Add indices in white text - zorder=3
        for i, p in padded_points.items():
            plt.text(
                p.theta, p.z, str(i), 
                color='white', 
                fontsize=font_size, 
                ha='center',        # Horizontal alignment
                va='center',        # Vertical alignment
                fontweight='bold',  # Makes it easier to read on blue
                zorder=3,
                clip_on = True
            )

    # Formatting
    plt.title(title, fontsize=20, pad=20)
    plt.xlabel('Theta (degrees)', fontsize=20)
    plt.ylabel('Z (Cartesian)', fontsize=20)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed visualization saved to: {save_path}")


if __name__ == "__main__":
    input_dir = Path("../input")
    output_dir = Path("../output")
    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_csv_path = input_dir / layout_file_name


    FB_vectors = generate_fibonacci_lattice(FB_count)
    FB_ztheta_points = [FB_vector.to_z_theta() for FB_vector in FB_vectors]


    if wrap or lift_bottom:
        FB_ztheta_points, padded_points = reorganize_points(FB_ztheta_points, layout_csv_path, wrap=wrap, lift_bottom=lift_bottom)
    else:
        padded_points = None

    connections = get_connections(
        layout_csv_path=layout_csv_path,
        columns=columns,
        rows=rows,
        wrap=wrap,
        lift_bottom=lift_bottom,
    )


    output_path = output_dir / output_file_name

    plot_z_theta_projection(
        FB_ztheta_points,
        padded_points=padded_points,
        title=graph_title,
        save_path=output_path,
        connections=connections,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize
    )

