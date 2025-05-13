from logging import getLogger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from tsppp.data.utils import sample_points
from tsppp.viz import visualize

logger = getLogger(__name__)


@hydra.main(config_path="config_eval", config_name="default")
def main(config):
    map_creator = hydra.utils.instantiate(config.map_creator)
    solver = hydra.utils.instantiate(config.solver)

    cfg = hydra.core.hydra_config.HydraConfig.get()
    map_type = cfg.runtime.choices["map_creator"]
    num_destinations = config.num_destinations
    graph_type = config.solver.graph_type
    logger.info(f"{map_type=}, {num_destinations=}, {graph_type=}")

    resultdir = (
        Path(config.resultdir)
        .joinpath(f"{map_type}_{num_destinations:03d}")
        .joinpath(config.solver.graph_type)
    )
    figdir = resultdir.joinpath("fig")
    figdir.mkdir(exist_ok=True, parents=True)

    stats = np.zeros((config.num_instances, 9))
    for i in range(config.num_instances):
        seed = config.seed + i
        occupancy_map = map_creator.create(seed)
        destinations = sample_points(occupancy_map, config.num_destinations, seed=seed)
        result = solver.solve(occupancy_map, destinations)
        if result.status:
            path = solver.execute(result)
            pathlen = path.get_length()
        else:
            path = np.zeros((0, 2))
            pathlen = -1

        time = result.get_total_time()
        stats[i] = [
            result.status,
            pathlen,
            len(solver.graph.G.nodes),
            len(solver.graph.G.edges),
            time,
            result.elapsed_time_graph,
            result.elapsed_time_cost,
            result.elapsed_time_tsp,
            seed,
        ]
        logger.info(f"{result.status=}, {pathlen=:.2f}, {time=:.2f}")
        fig, ax = plt.subplots(1, 1, figsize=[6, 6])
        visualize(occupancy_map, destinations, path, graph=solver.graph, ax=ax)
        fig.tight_layout()
        fig.savefig(figdir.joinpath(f"{i:04d}_{seed=}.png"))

    np.savetxt(resultdir.joinpath("stats.txt"), stats, delimiter=",")

    # Display summary if reference stats are given
    if config.reference_stats is not None:
        ref_stats = np.loadtxt(config.reference_stats, delimiter=",")
        assert np.all(stats[:, -1] == ref_stats[:, -1]), "Seed mismatch"

        sr = np.mean(stats[:, 0])
        spl = np.mean(
            stats[:, 0] * (ref_stats[:, 1] / np.maximum(ref_stats[:, 1], stats[:, 1]))
        )
        time = np.mean(stats[:, 4])

        logger.info(
            f"Summary (method: {graph_type}, ref: {config.reference_stats}) | {sr=:.2f}, {spl=:.2f}, {time=:.2f}"
        )


if __name__ == "__main__":
    main()
