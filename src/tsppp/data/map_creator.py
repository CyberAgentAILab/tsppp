from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skimage.io import imread
from skimage.transform import resize


@dataclass
class MapCreator:
    """
    A class that creates occupancy maps for robot route planning.

    Args:
        mode (str, optional): The mode for creating the occupancy map. Can be "random" or "image". Defaults to "random".
        image_dir (str, optional): The directory containing image files for creating occupancy maps. Defaults to None.
        map_size (int, optional): The size of the occupancy map. Defaults to 128.
        map_th (float, optional): The threshold value for converting images to occupancy maps. Defaults to 0.9.
        num_blobs (int, optional): The number of blobs in the occupancy map. Defaults to 10.
        blob_size (int, optional): The size of each blob in the occupancy map. Defaults to 10.

    Methods:
        __call__(seed: int) -> NDArray:
            Creates an occupancy map based on the specified mode.

    Attributes:
        mode (str): The mode for creating the occupancy map.
        image_dir (str): The directory containing image files for creating occupancy maps.
        map_size (int): The size of the occupancy map.
        map_th (float): The threshold value for converting images to occupancy maps.
        num_blobs (int): The number of blobs in the occupancy map.
        blob_size (int): The size of each blob in the occupancy map.
    """

    mode: str = "random"
    image_dir: str = None
    map_size: int = 128
    map_th: float = 0.5
    num_blobs: int = 10
    blob_size: int = 10

    def __post_init__(self):
        self.create = (
            self.load_occupancy_map
            if self.mode == "image"
            else self.create_occupancy_map
        )
        if self.image_dir is not None:
            self.image_files = list(Path(self.image_dir).glob("*.png"))
        else:
            self.image_files = []

    def __call__(self, seed: int) -> NDArray:
        return self.create(seed)

    def load_occupancy_map(self, seed: int) -> NDArray:
        """
        Loads an occupancy map from an image file.

        Args:
            seed (int): The seed for random number generation.

        Returns:
            NDArray: The loaded occupancy map.
        """
        assert len(self.image_files) > 0
        random_state = np.random.RandomState(seed)
        image_file = random_state.choice(self.image_files, 1)[0]
        image = imread(image_file, as_gray=True)
        # randomly crop the square part
        min_size = self.map_size
        image = resize(image, (min_size * 2, min_size * 2))
        rst = np.random.RandomState(seed)
        s1 = rst.randint(0, image.shape[0] - min_size + 1)
        s2 = rst.randint(0, image.shape[1] - min_size + 1)
        image = image[s1 : s1 + min_size, s2 : s2 + min_size]
        occupancy_map = resize(image, (self.map_size, self.map_size)) < self.map_th
        return occupancy_map.astype(np.float32)

    def create_occupancy_map(self, seed: int) -> NDArray:
        """
        Creates a random occupancy map.

        Args:
            seed (int): The seed for random number generation.

        Returns:
            NDArray: The created occupancy map.
        """
        random_state = np.random.RandomState(seed)
        occupancy_map = np.ones((self.map_size, self.map_size))
        occupancy_map[1:-1, 1:-1] = 0
        for _ in range(self.num_blobs):
            x = random_state.randint(0, self.map_size)
            y = random_state.randint(0, self.map_size)
            occupancy_map[x : x + self.blob_size, y : y + self.blob_size] = 1
        return occupancy_map.astype(np.float32)
