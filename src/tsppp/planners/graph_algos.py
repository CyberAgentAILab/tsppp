from logging import getLogger
from time import time
from typing import Callable, NamedTuple, Tuple

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import label
from scipy.spatial import cKDTree

from ..data.utils import create_input_image
from ..pipelines import CPRMPipeline, DDPM1DPipeline
from .graph_utils import check_validity, generate_boundary_nodes

logger = getLogger(__name__)


class Graph(NamedTuple):
    """
    Represents a graph.

    Attributes:
        G (nx.Graph): The graph object.
        node_tree (cKDTree): The KDTree object for efficient nearest neighbor search.
        occupancy_map (NDArray): The occupancy map representing the environment.
    """

    G: nx.Graph
    node_tree: cKDTree
    occupancy_map: NDArray


def get_graph_construction_algorithm(
    graph_type: str, seed: int = 0, checkpoint: str = None
) -> Callable:
    """
    Returns a graph construction algorithm based on the specified graph type.

    Args:
        graph_type (str): The type of graph to construct.
        seed (int, optional): The seed value for random number generation. Defaults to 0.
        checkpoint (str, optional): The path to the checkpoint file for pretrained models (if necessary). Defaults to None.

    Returns:
        Callable: A function that takes an occupancy map and destinations as input and returns a tuple containing the constructed graph and the execution time.
    """
    try:
        graph_type, N, param = graph_type.split("_")
    except:
        raise ValueError(f"Invalid graph type: {graph_type}")

    if graph_type == "random":
        assert (param == "r") or (
            param == "k"
        ), "the neighbor type should be either r or k."
        alg = lambda occ, wp: construct_random_graph(
            occ, wp, num_samples=int(N), neighbor_type=param, seed=seed
        )
    elif graph_type == "tspdiffuser":
        pipeline = DDPM1DPipeline.from_pretrained(checkpoint)
        alg = lambda occ, wp: construct_ddpm_graph(
            occ,
            wp,
            num_samples=int(N),
            pipeline=pipeline,
            num_neighbors=5,
            subsample_boundary_nodes=False,
            num_inference_steps=int(param),
            seed=seed,
        )
    elif graph_type == "tspdiffuser+":
        pipeline = DDPM1DPipeline.from_pretrained(checkpoint)
        alg = lambda occ, wp: construct_ddpm_graph(
            occ,
            wp,
            num_samples=int(N),
            pipeline=pipeline,
            num_neighbors=15,
            subsample_boundary_nodes=True,
            num_inference_steps=int(param),
            seed=seed,
        )
    elif graph_type == "diffuser":
        pipeline = DDPM1DPipeline.from_pretrained(checkpoint)
        alg = lambda occ, wp: construct_ddpm_graph(
            occ,
            wp,
            num_samples=1,
            pipeline=pipeline,
            num_neighbors=5,
            num_inference_steps=int(param),
            add_boundary_nodes=True,
            seed=seed,
        )
    elif graph_type == "cprm":
        pipeline = CPRMPipeline.from_pretrained(checkpoint)
        alg = lambda occ, wp: construct_cprm_graph(
            occ,
            wp,
            num_samples=int(N),
            pipeline=pipeline,
            lambda_value=float(param),
            seed=seed,
        )
    else:
        raise ValueError(f"Invalid graph type: {graph_type}")

    def run(occupancy_map: NDArray, destinations: NDArray) -> Tuple[Graph, float]:
        """
        Constructs a graph based on the specified graph type.

        Args:
            occupancy_map (NDArray): The occupancy map.
            destinations (NDArray): The destinations.

        Returns:
            Tuple[Graph, float]: A tuple containing the constructed graph and the execution time.
        """
        tic = time()
        output = alg(occupancy_map, destinations)
        toc = time() - tic
        return output, toc

    return run


def construct_random_graph(
    occupancy_map: NDArray,
    destinations: NDArray,
    num_samples: int,
    neighbor_type: str = "r",
    seed: int = 0,
) -> Graph:
    """
    Constructs a random graph representation of the environment.

    Args:
        occupancy_map (NDArray): The occupancy map of the environment.
        destinations (NDArray): The destinations to be used for constructing the graph.
        num_samples (int): The number of nodes to generate in the graph.
        neighbor_type (str, optional): The type of neighbor connection to use. Can be "r" for radius or "k" for k-nearest neighbors. Defaults to "r".
        seed (int, optional): The random seed for reproducibility. Defaults to 0.

    Returns:
        Graph: The constructed graph.
    """

    map_size = occupancy_map.shape[-1]
    rst = np.random.RandomState(seed)
    nodes = []
    while len(nodes) < num_samples:
        node_candidate = rst.rand(2) * map_size
        node_int = np.clip(node_candidate.astype(int), 0, map_size - 1)
        if occupancy_map[node_int[0], node_int[1]] == 0:
            nodes.append(node_candidate)
    nodes = np.vstack(nodes)
    G = nx.Graph()
    G.add_nodes_from([tuple(node) for node in nodes])

    node_tree = cKDTree(nodes)
    num_samples = len(nodes)
    # neighbor parameters are determined based on https://arxiv.org/pdf/1105.1186.pdf
    if neighbor_type == "r":
        gamma = 1.0  # worked empirically well
        param = (
            gamma * np.sqrt(np.log(num_samples) / num_samples) * occupancy_map.shape[0]
        )
    elif neighbor_type == "k":
        param = int(np.e * (1 + 1 / 2.0) * np.log(num_samples))
    else:
        raise ValueError(f"Invalid neighbor_type: {neighbor_type}")

    for node in nodes:
        if neighbor_type == "r":
            n_indices = node_tree.query_ball_point(node, param)
        elif neighbor_type == "k":
            n_indices = node_tree.query(node, k=param)[1]
        for n_ind in n_indices:
            is_valid, dist = check_validity(node, nodes[n_ind], occupancy_map)
            if is_valid & (tuple(node) != tuple(nodes[n_ind])):
                G.add_edge(tuple(node), tuple(nodes[n_ind]), weight=dist)

    return Graph(G=G, node_tree=node_tree, occupancy_map=occupancy_map)


def construct_ddpm_graph(
    occupancy_map: NDArray,
    destinations: NDArray,
    num_samples: int,
    pipeline: DDPM1DPipeline,
    num_inference_steps: int,
    num_neighbors: int = 5,
    guidance_type: str = "dt_10",
    add_boundary_nodes: bool = True,
    subsample_boundary_nodes: bool = False,
    seed: int = 0,
) -> Graph:
    """
    Constructs a graph representation for robot route planning using the DDPM algorithm.

    Args:
        occupancy_map (NDArray): The occupancy map of the environment.
        destinations (NDArray): The destinations for the robot to visit.
        num_samples (int): The number of samples to generate for each destination.
        pipeline (DDPM1DPipeline): The DDPM pipeline for generating samples.
        num_inference_steps (int): The number of inference steps to perform.
        num_neighbors (int, optional): The number of neighboring nodes to connect each node to. Defaults to 5.
        guidance_type (str, optional): The type of guidance to use during sampling. Defaults to "dt_10".
        add_boundary_nodes (bool, optional): Whether to add boundary nodes to the graph. Defaults to True.
        subsample_boundary_nodes (bool, optional): Whether to subsample the boundary nodes. Defaults to False.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.

    Returns:
        Graph: The constructed graph representation.
    """
    torch.manual_seed(seed)
    map_size = occupancy_map.shape[-1]
    image = create_input_image(
        occupancy_map, destinations, apply_distance_transform=True
    )
    destinations = torch.from_numpy(destinations.T).unsqueeze(0)
    image = torch.from_numpy(image).unsqueeze(0)
    samples = pipeline(
        destinations,
        image,
        num_inference_steps=num_inference_steps,
        num_samples=num_samples,
        guidance_type=guidance_type,
        is_trimmed=True,
    ).samples
    nodes = (samples.transpose(0, 2, 1).reshape(-1, 2) + 1) * map_size / 2

    # detect collided obstacles
    labeled_map = label(occupancy_map)[0]
    hit_id = np.unique(labeled_map[nodes.astype(int)[:, 0], nodes.astype(int)[:, 1]])
    if len(hit_id) > 1:
        hit_id = hit_id[1:]
        collided_obs = np.any(np.dstack([labeled_map == i for i in hit_id]), axis=2)
    else:
        collided_obs = np.zeros_like(occupancy_map)

    nodes = nodes[occupancy_map[nodes.astype(int)[:, 0], nodes.astype(int)[:, 1]] == 0]

    G = nx.Graph()
    G.add_nodes_from([tuple(node) for node in nodes])

    # edges along paths
    r_max = np.linalg.norm(np.diff(samples, axis=2), axis=1).max() * map_size / 2
    for i in range(len(nodes)):
        is_valid, dist = check_validity(nodes[i], nodes[i - 1], occupancy_map)
        if is_valid & (dist <= r_max):
            G.add_edge(tuple(nodes[i]), tuple(nodes[i - 1]), weight=dist)

    if add_boundary_nodes:
        boundary_nodes = generate_boundary_nodes(collided_obs)
        if subsample_boundary_nodes:
            boundary_nodes = boundary_nodes[0:-1:5]
        G.add_nodes_from([tuple(node) for node in boundary_nodes])
        nodes = np.vstack((nodes, boundary_nodes))

    # add destination nodes to make connection rule consistent to be k-NN
    destination_nodes = destinations[0].T.numpy()
    G.add_nodes_from([tuple(node) for node in destination_nodes])
    nodes = np.vstack((nodes, destination_nodes))
    node_tree = cKDTree(nodes)

    for node in nodes:
        # edges between paths
        n_indices = node_tree.query(node, k=num_neighbors)[1]
        for n_ind in n_indices:
            is_valid, dist = check_validity(node, nodes[n_ind], occupancy_map)
            if is_valid & (tuple(node) != tuple(nodes[n_ind])):
                G.add_edge(tuple(node), tuple(nodes[n_ind]), weight=dist)

    return Graph(G=G, node_tree=node_tree, occupancy_map=occupancy_map)


def construct_cprm_graph(
    occupancy_map: NDArray,
    destinations: NDArray,
    num_samples: int,
    pipeline: CPRMPipeline,
    lambda_value: float,
    seed: int = 0,
) -> Graph:
    """
    Constructs a graph for the CPRM (Critical Point Robot Motion) algorithm.

    Args:
        occupancy_map (NDArray): The occupancy map of the environment.
        destinations (NDArray): The destinations for the robot to navigate.
        num_samples (int): The total number of samples to generate for the graph.
        pipeline (CPRMPipeline): The pipeline for generating critical samples.
        lambda_value (float): The lambda value used to determine the number of critical samples.
        seed (int, optional): The seed value for random number generation. Defaults to 0.

    Returns:
        Graph: The constructed graph for the CPRM algorithm.
    """
    torch.manual_seed(seed)
    image = create_input_image(
        occupancy_map, destinations, apply_distance_transform=True
    )
    image = torch.from_numpy(image).unsqueeze(0)
    destinations = torch.from_numpy(destinations.T).unsqueeze(0)

    num_critical_samples = int(
        np.log(num_samples) * lambda_value
    )  # see Sec V.A of https://arxiv.org/pdf/1910.03701.pdf

    # uniform nodes
    rst = np.random.RandomState(seed)
    uniform_nodes = []
    map_size = occupancy_map.shape[-1]
    while len(uniform_nodes) < num_samples - num_critical_samples:
        node_candidate = rst.rand(2) * map_size
        node_int = np.clip(node_candidate.astype(int), 0, map_size - 1)
        if occupancy_map[node_int[0], node_int[1]] == 0:
            uniform_nodes.append(node_candidate)
    uniform_nodes = np.vstack(uniform_nodes)
    G = nx.Graph()
    G.add_nodes_from([tuple(node) for node in uniform_nodes])
    uniform_tree = cKDTree(uniform_nodes)
    num_samples = len(uniform_nodes)
    r = (
        np.sqrt(np.log(num_samples) / num_samples) * occupancy_map.shape[0]
    )  # see Sec 3.3 of https://arxiv.org/pdf/1105.1186.pdf

    # connect non-critical nodes locally
    for node in uniform_nodes:
        n_indices = uniform_tree.query_ball_point(node, r)
        for n_ind in n_indices:
            is_valid, dist = check_validity(node, uniform_nodes[n_ind], occupancy_map)
            if is_valid & (tuple(node) != tuple(uniform_nodes[n_ind])):
                G.add_edge(tuple(node), tuple(uniform_nodes[n_ind]), weight=dist)

    # critical nodes
    samples = pipeline(destinations, image, num_samples=num_critical_samples).samples
    critical_nodes = samples[0].T  # (num_samples, 2)
    all_nodes = np.vstack((uniform_nodes, critical_nodes))
    node_tree = cKDTree(all_nodes)

    G.add_nodes_from([tuple(node) for node in critical_nodes])

    # connect critical nodes to all nodes
    for critical_node in critical_nodes:
        n_indices = node_tree.query_ball_point(node, r)
        for n_ind in n_indices:
            # for node in all_nodes:
            node = all_nodes[n_ind]
            is_valid, dist = check_validity(critical_node, node, occupancy_map)
            if is_valid & (tuple(critical_node) != tuple(node)):
                G.add_edge(
                    tuple(critical_node), tuple(node), weight=dist
                )  # original implementation added critical nodes here

    return Graph(G=G, node_tree=node_tree, occupancy_map=occupancy_map)


def expand_graph(graph: Graph, node: tuple) -> Graph:
    """
    Expands the given graph by adding a new node and connecting it to nearby nodes if they are valid.

    Args:
        graph (Graph): The graph to be expanded.
        node (tuple): The new node to be added to the graph.

    Returns:
        Graph: The expanded graph.
    """
    # check if the node is already in the graph
    if node in graph.G:
        return graph
    else:
        G = graph.G
        node_tree = graph.node_tree
        occupancy_map = graph.occupancy_map
        G.add_node(node)
        nodes = node_tree.data
        num_nodes = len(nodes)
        r = np.sqrt(np.log(num_nodes) / num_nodes) * occupancy_map.shape[0]
        n_indices = graph.node_tree.query_ball_point(node, r)
        for n_ind in n_indices:
            is_valid, dist = check_validity(np.array(node), nodes[n_ind], occupancy_map)
            if is_valid:
                G.add_edge(node, tuple(nodes[n_ind]), weight=dist)

    return Graph(G=G, node_tree=node_tree, occupancy_map=occupancy_map)
