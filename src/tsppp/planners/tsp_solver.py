from dataclasses import dataclass
from time import time
from typing import NamedTuple, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .graph_algos import Graph, expand_graph, get_graph_construction_algorithm

LARGE_NUMBER = 1e9


class Path(NamedTuple):
    data: NDArray
    status: bool

    def __len__(self):
        # return len(self.data)
        return int(self.get_length())

    def __add__(self, other):
        return Path(np.vstack([self.data, other.data]), self.status and other.status)

    def __getitem__(self, indices):
        return self.data[indices]

    def get_length(self):
        try:
            length = np.linalg.norm(np.diff(self.data, axis=0), axis=-1).sum()
        except:
            length = -1
        return length

    def __repr__(self) -> str:
        return f"Path status: {self.status}, length: {self.get_length():0.2f}"


class TSPResult(NamedTuple):
    graph_type: str
    graph: Graph
    destinations: NDArray
    cost_matrix: NDArray
    tour: NDArray
    status: bool
    elapsed_time_graph: float
    elapsed_time_cost: float
    elapsed_time_tsp: float

    def __repr__(self) -> str:
        return f"TSP status ({self.graph_type}): {self.status}, total_time: {self.get_total_time():0.2e} (graph: {self.elapsed_time_graph:0.2e}, cost: {self.elapsed_time_cost:0.2e}, tsp: {self.elapsed_time_tsp:0.2e})"

    def get_total_time(self) -> float:
        return self.elapsed_time_graph + self.elapsed_time_cost + self.elapsed_time_tsp


@dataclass
class TSPSolver:
    """
    TSP Solver

    Attributes:
        graph_type (str): The type of graph construction algorithm to use. Defaults to "random_3000_r".
        seed (int): The seed for the random number generator. Defaults to 0.
        checkpoint (str): The path to the checkpoint file to load the graph from.

    Notes:
        grapy_type can be one of the following:
        - "random_N_r": Random graph with N nodes, where r for the r-neighbor is determined automatically based on N (e.g., "random_3000_r").
        - "tspdiffuser_N_K": DDPM graph consisting of N paths generated with 5 denoising steps and connected with K neighbors (e.g., "tspdiffuser_10_5").
        - "tspdiffuser+_N_K": tsp diffuser with subsampled boundary nodes, which is useful when K is large (e.g., "tspdiffuser+_10_15").
        - "diffuser_1_I": Diffuser graph consisting of a single path generated with I denoising steps and K=15 (e.g., "diffuser_10_5").
        - "cprm_N_L": CPRM graph consisting of N samples in total, where (log(N) * L) nodes are sampled from the learned model (e.g., "cprm_100_10").
    """

    graph_type: str = "random_3000_r"
    seed: int = 0
    checkpoint: str = None

    def __post_init__(self):
        self.construct_graph = get_graph_construction_algorithm(
            self.graph_type, self.seed, self.checkpoint
        )

    def solve(self, occupancy_map: NDArray, destinations: NDArray) -> TSPResult:
        """
        Runs the robot route planning algorithm.

        Args:
            occupancy_map (NDArray): The occupancy map.
            destinations (NDArray): Array of destinations representing the locations to visit.

        Returns:
            TSPResult: Result of the robot route planning algorithm.
        """

        # reset graph
        graph, elapsed_time_graph = self.construct_graph(occupancy_map, destinations)
        self.graph = graph

        cost_matrix, elapsed_time_cm = self.construct_cost_matrix(destinations, graph)
        tour, elapsed_time_tsp = self.solve_metric_tsp(cost_matrix)

        solved = LARGE_NUMBER not in [
            cost_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)
        ]

        tsp_result = TSPResult(
            self.graph_type,
            graph,
            destinations,
            cost_matrix,
            tour,
            solved,
            elapsed_time_graph,
            elapsed_time_cm,
            elapsed_time_tsp,
        )
        return tsp_result

    def execute(self, tsp_result: TSPResult) -> Path:
        """
        Executes the robot route planning algorithm.

        Args:
            tsp_result (TSPResult): Result of the robot route planning algorithm.

        Returns:
            Path: The path to be followed by the robot.
        """
        destinations = tsp_result.destinations[tsp_result.tour]
        N = len(destinations)
        path = None
        for i in range(N - 1):
            path_segment = self.get_path(
                destinations[i], destinations[i + 1], tsp_result.graph
            )
            if not path_segment.status:
                return Path(None, False)
            else:
                path = path_segment if path is None else path + path_segment

        return path

    def get_path(
        self,
        src_pos: NDArray,
        dst_pos: NDArray,
        graph: Graph,
    ) -> Path:
        """
        Calculates the shortest path between the source position and the destination position
        using A* algorithm.

        Args:
            src_pos (NDArray): The source position.
            dst_pos (NDArray): The destination position.
            graph (Graph): The graph representing the environment.

        Returns:
            Path: The shortest path between the source and destination positions.
        """

        src_node = tuple(src_pos)
        graph = expand_graph(graph, src_node)
        dst_node = tuple(dst_pos)
        graph = expand_graph(graph, dst_node)

        try:

            def euc_dist(a, b):
                # Implementation of Euclidean dist function
                (x1, y1) = a
                (x2, y2) = b
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            path = nx.astar_path(
                graph.G, src_node, dst_node, heuristic=euc_dist, weight="weight"
            )

        except nx.NetworkXNoPath:
            return Path(data=None, status=False)

        return Path(data=np.asarray(path), status=True)

    def construct_cost_matrix(
        self, destinations: NDArray, graph: Graph
    ) -> Tuple[NDArray, float]:
        """
        Constructs a cost matrix based on the given destinations.

        Parameters:
            destinations (NDArray): An array of destinations.
            graph (Graph): The graph representing the environment.

        Returns:
            Tuple[NDArray, float]: A tuple containing the cost matrix and the elapsed time.
        """
        N = len(destinations)
        cost_matrix = np.zeros((N, N), np.int32)
        tic = time()
        for i in range(N):
            for j in range(i, N):
                if i != j:
                    path = self.get_path(destinations[i], destinations[j], graph)
                    cost = len(path) if path.status else LARGE_NUMBER
                    cost_matrix[i, j] = cost_matrix[j, i] = cost
        elapsed_time = time() - tic

        assert cost_matrix.dtype == np.int32

        return cost_matrix, elapsed_time

    def solve_metric_tsp(self, cost_matrix: NDArray) -> Tuple[NDArray, float]:
        """
        Solves the metric Traveling Salesman Problem (TSP) using the given cost matrix.
        Copied from or-tools' example at https://developers.google.com/optimization/routing/tsp

        Args:
            cost_matrix (NDArray): The cost matrix representing the distances between nodes.

        Returns:
            Tuple[NDArray, float]: A tuple containing the optimal tour and the elapsed time in seconds.
                - The optimal tour is a list of node indices representing the order of nodes to visit.
                - The elapsed time is the time taken to solve the problem.
        """

        tic = time()
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(cost_matrix), 1, 0)

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            return cost_matrix[manager.IndexToNode(from_index)][
                manager.IndexToNode(to_index)
            ]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        elapsed_time = time() - tic

        if solution:
            index = routing.Start(0)
            tour = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                tour.append(manager.IndexToNode(index))
            tour = np.asarray(tour)
        else:
            tour = None

        return tour, elapsed_time
