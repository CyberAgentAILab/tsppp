from dataclasses import dataclass
from logging import getLogger

from ..planners.tsp_solver import TSPSolver
from .map_creator import MapCreator
from .utils import padding, sample_points

logger = getLogger(__name__)


@dataclass
class DataGenerator:
    """
    Generates data for TSPPP

    Attributes:
        tsp_solver (TSPSolver): The solver used for solving the Traveling Salesperson Problem.
        map_creator (MapCreator): The creator used for generating occupancy maps.
        num_destinations (int): The number of destinations to generate.
        dilation_factor (int): The dilation factor for generating destinations.
        max_trials (int): The maximum number of trials to attempt for generating valid data.
        return_traj (bool): Whether to return the trajectory along with the data.
        max_valid_length (int): The maximum valid length of the data.
        num_skipframes (int): The number of frames to skip when padding the data.

    Methods:
        generate(seed: int) -> dict:
            Generates a single data sample.

    """

    tsp_solver: TSPSolver
    map_creator: MapCreator
    num_destinations: int = 10
    dilation_factor: int = 5
    max_trials: int = 5
    max_valid_length: int = 256
    num_skipframes: int = 5

    def __str__(self) -> str:
        return f"random_ms{self.map_creator.map_size}_nb{self.map_creator.num_blobs}_bs{self.map_creator.blob_size}_nw{self.num_destinations}_{self.tsp_solver.graph_type}"

    def generate(self, seed: int) -> dict:
        """
        Generates a single data sample.

        Args:
            seed (int): The seed value for generating random data.

        Returns:
            dict: A dictionary containing the generated destinations, data, and occupancy map.

        """
        occupancy_map = self.map_creator.create(seed)

        sample = {
            "destinations": None,
            "data": None,
            "occupancy_map": occupancy_map.astype(int),
        }
        for n in range(self.max_trials):
            destinations = sample_points(
                occupancy_map, self.num_destinations, self.dilation_factor, seed + n
            )
            result = self.tsp_solver.solve(occupancy_map, destinations)
            if not result.status:
                logger.warning(
                    f"Failed to solve TSP. Retry ({n + 1}/{self.max_trials})"
                )
                continue
            output = self.tsp_solver.execute(result)
            if not output.status:
                logger.warning(
                    f"Failed to execute TSP solution. Retry ({n + 1}/{self.max_trials})"
                )
                continue
            data = output.data[0 :: self.num_skipframes]
            data, validity = padding(data, self.max_valid_length)
            if not validity:
                continue

            return {
                "destinations": destinations,
                "data": data,
                "occupancy_map": occupancy_map.astype(int),
            }

        # If the maximum number of trials is reached, return the empty list.
        return sample
