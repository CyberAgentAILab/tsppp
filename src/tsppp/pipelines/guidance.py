from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@dataclass
class BaseGuidance:
    """
    Base class for guidance algorithms.
    """

    occupancy_map: torch.tensor
    destinations: torch.tensor


def get_guidance(
    guidance_type: str, occupancy_map: torch.tensor, destinations: torch.tensor
) -> BaseGuidance:
    """
    Factory function to create a guidance object based on the specified type.

    Args:
        guidance_type (str): The type of guidance algorithm.
        occupancy_map (torch.tensor): The occupancy map.
        destinations (torch.tensor): The destinations.

    Returns:
        BaseGuidance: The guidance object.

    Raises:
        NotImplementedError: If the specified guidance type is not implemented.
    """
    guidance_type, param = guidance_type.split("_")
    if guidance_type == "dt":
        return DTGuidance(occupancy_map, destinations, num_interp_steps=int(param))
    else:
        raise NotImplementedError(f"Guidance type {guidance_type} not implemented")


@dataclass
class DTGuidance(BaseGuidance):
    """
    Distance Transform (DT) guidance algorithm.

    Args:
        BaseGuidance: The base guidance class.
        num_interp_steps (int): The number of interpolation steps.

    Attributes:
        obs_gradmap (torch.tensor): The gradient map of the occupancy map.
        map_size (int): The size of the map.
    """

    num_interp_steps: int = 10

    def __post_init__(self):
        assert self.occupancy_map.ndim == 2
        map_size = self.occupancy_map.shape[0]
        obs_indices = distance_transform_edt(
            np.asarray(self.occupancy_map.cpu()),
            return_distances=False,
            return_indices=True,
        )
        locs = np.stack(np.meshgrid(np.arange(map_size), np.arange(map_size)))[::-1]

        self.obs_gradmap = torch.tensor(
            (obs_indices - locs) / map_size, device=self.occupancy_map.device
        )

        self.map_size = map_size

    def __call__(self, samples: torch.tensor) -> torch.tensor:
        """
        Perform the guidance algorithm on the given samples.

        Args:
            samples (torch.tensor): The input samples.

        Returns:
            torch.tensor: The output samples.
        """
        samples_interp = []
        for sample in samples.cpu():
            sample = np.vstack(
                [
                    np.interp(
                        np.linspace(0, 1, len(s) * self.num_interp_steps),
                        np.linspace(0, 1, len(s)),
                        s,
                    )
                    for s in sample
                ]
            )
            samples_interp.append(sample)
        samples_interp = torch.tensor(
            np.stack(samples_interp), device=samples.device
        ).float()
        samples_int = ((samples_interp + 1) * self.map_size / 2).int()
        samples_int = torch.clamp(samples_int, 0, self.map_size - 1)
        obs_grads = self.obs_gradmap[:, samples_int[:, 0, :], samples_int[:, 1, :]]
        obs_grads = obs_grads.permute(
            1, 0, 2
        )  # (2, batch_size, T) -> (batch_size, 2, T)
        samples_interp = samples_interp + obs_grads.float()
        samples_output = []
        for sample in samples_interp.cpu():
            sample = np.vstack(
                [
                    np.interp(
                        np.linspace(0, 1, len(s) // self.num_interp_steps),
                        np.linspace(0, 1, len(s)),
                        s,
                    )
                    for s in sample
                ]
            )
            samples_output.append(sample)
        samples_output = torch.tensor(
            np.stack(samples_output), device=samples.device
        ).float()
        return samples_output
