import numpy as np
import pytest


@pytest.fixture
def instance():
    from tsppp.data.map_creator import MapCreator
    from tsppp.data.utils import sample_points

    map_creator = MapCreator(map_size=128, num_blobs=10, blob_size=10)
    occupancy_map = map_creator.create(0)
    destinations = sample_points(occupancy_map, 10, seed=0)

    return occupancy_map, destinations


def test_check_validity(instance):
    from tsppp.planners.graph_utils import check_validity

    occupancy_map, destinations = instance
    validity, dist = check_validity(np.array([1, 1]), np.array([2, 1]), occupancy_map)
    assert validity & np.isclose(dist, 1)

    validity, dist = check_validity(
        np.array([1, 1]), np.array([127, 127]), occupancy_map
    )
    assert not validity & np.isclose(dist, np.linalg.norm([126, 126]))


def test_random_graph(instance):
    from tsppp.planners.graph_algos import construct_random_graph

    occupancy_map, destinations = instance
    graph = construct_random_graph(occupancy_map, destinations, num_samples=100, seed=0)
    assert len(graph.G.nodes) == len(graph.node_tree.data) > 0


def test_prm_solver(instance):
    from tsppp.planners.tsp_solver import LARGE_NUMBER, TSPSolver

    occupancy_map, destinations = instance
    solver = TSPSolver(graph_type="random_100_r", seed=0)
    graph, elapsed_time_graph = solver.construct_graph(occupancy_map, destinations)
    path = solver.get_path([1, 1], [126, 126], graph)
    assert path.status

    solver = TSPSolver(graph_type="random_100_r", seed=0)
    result = solver.solve(occupancy_map, destinations)
    assert result.status
    assert result.cost_matrix.shape == (10, 10)

    occupancy_map[:, 32:64] = 1
    destinations = np.array([[10, 10], [100, 100]])
    solver = TSPSolver(graph_type="random_100_r", seed=0)
    result = solver.solve(occupancy_map, destinations)
    assert not result.status
    assert result.cost_matrix[1, 0] == LARGE_NUMBER
