import numpy as np


def test_map_creator():
    from tsppp.data.map_creator import MapCreator  # first import takes some time

    map_creator = MapCreator(map_size=128, num_blobs=10, blob_size=10)
    occupancy_map = map_creator.create(0)
    assert occupancy_map.dtype == np.float32
    assert occupancy_map.shape == (128, 128)
    assert (occupancy_map.max() == 1) & (occupancy_map.min() == 0)


def test_sampler():
    from tsppp.data.utils import sample_points

    occupancy_map = np.zeros((128, 128))
    occupancy_map[32:] = 1
    points = sample_points(occupancy_map, 100, seed=0)
    assert points.dtype == np.float32
    assert points.shape == (100, 2)

    # check if all samples are in the free space
    assert np.all([occupancy_map[int(p[0]), int(p[1])] == 0 for p in points])
