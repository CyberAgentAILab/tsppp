import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from numpy.typing import NDArray
from tsppp.planners.graph_algos import Graph


# visualize result
def visualize(
    occupancy_map: NDArray,
    destinations: NDArray,
    path: NDArray = None,
    pred_path: NDArray = None,
    graph: Graph = None,
    ax: plt.Axes = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if graph is not None:
        nx.draw(
            graph.G,
            pos={tuple(x): x for x in graph.G.nodes},
            node_size=1,
            alpha=0.2,
            node_color="gray",
            edge_color="gray",
            ax=ax,
        )
    ax.imshow(occupancy_map.T, cmap="Blues", vmax=1.5, vmin=-0.05)
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "-", color="k", lw=2, label="gt-path")
    if pred_path is not None:
        ax.plot(
            pred_path[:, 0],
            pred_path[:, 1],
            "-",
            color=cm.Set1(2),
            lw=2,
            label="pred-path",
            zorder=5,
        )
    ax.scatter(
        destinations[:, 0],
        destinations[:, 1],
        80,
        color=cm.Set1(0),
        edgecolor="k",
        lw=2,
        zorder=10,
        label="destinations",
    )

    ax.axis("off")
    ax.set_xlim([0, occupancy_map.shape[0]])
    ax.set_ylim([occupancy_map.shape[1], 0])
