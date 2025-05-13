import math
from dataclasses import dataclass
from logging import getLogger

import dask.dataframe as dd
import numpy as np
from numpy.typing import NDArray
from skimage.morphology import dilation
from torch.utils.data import Dataset

from .utils import create_input_image

logger = getLogger(__name__)


@dataclass
class DDPMDataset(Dataset):
    """
    A dataset class for training DDPM

    Attributes:
        data_file (str): The path to the dataset file.
        apply_distance_transform (bool): Flag indicating whether to apply distance transform to the data.

    """

    data_file: str
    apply_distance_transform: bool = True

    def __post_init__(self):
        dask_df = dd.read_parquet(self.data_file, dtype_backend="pyarrow")
        # Assuming 'dask_array' is your Dask array
        dask_array = dask_df.to_dask_array(lengths=True)
        dask_array = dask_array.persist()  # persist array in memory
        self.dask_array = dask_array

    def __len__(self):
        return len(self.dask_array)

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple[NDArray, NDArray, NDArray]: A tuple containing the normalized path, normalized destinations, and input image.

        """
        sample = self.dask_array[idx].compute()
        map_size = int(math.sqrt(len(sample[2])))
        destinations, data, occupancy_map = (
            sample[0].reshape(-1, 2),
            sample[1].reshape(-1, 2),
            sample[2].reshape(map_size, map_size),
        )
        assert len(destinations) < len(data)

        # normalize the sample into [-1, 1]
        data_normalized = (data - map_size / 2) / (map_size / 2)
        destinations_normalized = (destinations - map_size / 2) / (map_size / 2)
        image = create_input_image(
            occupancy_map, destinations, self.apply_distance_transform
        )

        return (
            data_normalized.astype(np.float32).T,
            destinations_normalized.astype(np.float32).T,
            image.astype(np.float32),
        )


@dataclass
class CPRMDataset(DDPMDataset):
    """
    Dataset class for training CPRM

    Attributes:
        dilation_size (int): The size of dilation for the data image.

    """

    dilation_size: int = 9

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Get the item at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple[NDArray, NDArray, NDArray]: A tuple containing the path image, normalized destinations, and input image.
        """
        data, destinations, image = super().__getitem__(idx)
        data_image = np.zeros_like(image[0])
        map_size = image.shape[-1]
        data_unnormalized = (data + 1) * map_size / 2
        data_interp = np.vstack(
            [
                np.interp(np.linspace(0, 1, len(d) * 100), np.linspace(0, 1, len(d)), d)
                for d in data_unnormalized
            ]
        ).astype(int)
        data_image[data_interp[0], data_interp[1]] = 1
        if self.dilation_size > 0:
            data_image = dilation(
                data_image, np.ones((self.dilation_size, self.dilation_size))
            )
        data_image[image[0] == 1] = 0

        return data_image, destinations, image
