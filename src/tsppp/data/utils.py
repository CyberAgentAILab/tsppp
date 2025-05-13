from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_cdt, label
from skimage.morphology import dilation


def dilate(occupancy_map: NDArray, dilation_factor: int) -> NDArray:
    """
    Dilates the given occupancy map by the specified factor.

    Args:
        occupancy_map (NDArray): The occupancy map to dilate.
        dilation_factor (int): The factor by which to dilate the occupancy map.

    Returns:
        NDArray: The dilated occupancy map.
    """
    return dilation(occupancy_map, np.ones((dilation_factor, dilation_factor)))


def sample_points(
    occupancy_map: NDArray, N: int, dilation_factor: int = 5, seed: int = 0
) -> NDArray:
    """
    Samples N points from the largest free region in the occupancy map

    Args:
        occupancy_map (NDArray): The occupancy map.
        N (int): The number of points to sample.
        dilation_factor (int): The factor by which to dilate the occupancy map. Defaults to 5.
        seed (int, optional): The seed for the random number generator. Defaults to 0.

    Returns:
        NDArray: The sampled points.
    """
    # Identify the non-obstacle areas
    rst = np.random.RandomState(seed)

    # avoid sampling from edges
    edge_map = np.zeros_like(occupancy_map)
    edge_map[:1, :] = 1
    edge_map[:, :1] = 1
    edge_map[-1:, :] = 1
    edge_map[:, -1:] = 1

    occ = dilate(np.maximum(occupancy_map, edge_map), dilation_factor)

    # Identify largest valid area
    labeled_occ, num_valid = label(occ == 0)
    areas = [(labeled_occ == i).sum() for i in range(1, num_valid + 1)]
    largest_area = labeled_occ == np.argmax(areas) + 1

    valid_pos = np.vstack(np.where(largest_area)).T

    # Randomly sample N indices
    sampled_indices = rst.choice(len(valid_pos), N, replace=False)

    # Get the sampled points
    sampled_points = valid_pos[sampled_indices].astype(np.float32)

    return sampled_points


def padding(sample: NDArray, max_valid_length: int) -> Tuple[NDArray, bool]:
    """
    Circular-pad the input sample to a specified length

    Args:
        sample (NDArray): The input sample to be padded.
        max_valid_length (int): The maximum valid length of the sample.

    Returns:
        Tuple[NDArray, bool]: The padded sample and a boolean indicating if padding was applied.
    """

    if len(sample) > max_valid_length:
        return sample[:max_valid_length], False

    # circular padding
    sample = np.pad(
        sample,
        ((0, max_valid_length - len(sample)), (0, 0)),
        mode="wrap",
    )
    return sample, True


def create_input_image(
    occupancy_map: NDArray, destinations: NDArray, apply_distance_transform: bool = True
) -> NDArray:
    """
    Create an input image that stacks the occupancy map and the destinations.

    Args:
        occupancy_map (NDArray): The occupancy map of the environment.
        destinations (NDArray): The destinations for the robot to navigate.
        apply_distance_transform (bool, optional): Whether to apply distance transform.
            Defaults to True.

    Returns:
        NDArray: The input image for diffusion models.
    """
    map_size = occupancy_map.shape[-1]
    destination_map = np.zeros_like(occupancy_map)
    destination_map[destinations[:, 0].astype(int), destinations[:, 1].astype(int)] = -1

    image = np.stack([occupancy_map, destination_map], axis=0)
    # apply distance transform
    if apply_distance_transform:
        occupancy_map_d = distance_transform_cdt(1 - occupancy_map) / map_size
        destination_map_d = -distance_transform_cdt(1 + destination_map) / map_size
        image_d = np.stack([occupancy_map_d, destination_map_d], axis=0)
        image = np.concatenate([image, image_d], axis=0)

    return image.astype(np.float32)
