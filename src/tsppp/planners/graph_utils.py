from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_cdt


def generate_boundary_nodes(occupancy_map: NDArray) -> NDArray:
    """
    Generates boundary nodes from the given occupancy map.

    Parameters:
        occupancy_map (NDArray): The occupancy map representing the environment.

    Returns:
        NDArray: The boundary nodes of the occupancy map.
    """
    dt = distance_transform_cdt(1 - occupancy_map)
    boundary_nodes = np.argwhere(dt == 1)
    return boundary_nodes


def check_validity(
    a: NDArray, b: NDArray, occupancy_map: NDArray, step: float = 0.5
) -> Tuple[bool, float]:
    """
    Check the validity of a path between points 'a' and 'b' on an occupancy map.

    Parameters:
        a (NDArray): Starting point coordinates.
        b (NDArray): Ending point coordinates.
        occupancy_map (NDArray): Occupancy map representing obstacles.
        step (float, optional): Step size for checking validity. Defaults to 0.5.

    Returns:
        Tuple[bool, float]: A tuple containing a boolean indicating path validity and the distance between 'a' and 'b'.
    """

    dist = np.linalg.norm(b - a)
    uv = (b - a) / (dist + 1e-10)
    num_steps = (dist // step).astype(int)
    for i in range(num_steps):
        vint = (a + uv * i * step).astype(int)
        if occupancy_map[vint[0], vint[1]] == 1:
            return False, dist

    return True, dist
