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


def test_cprm_unet(instance):
    import torch
    from tsppp.data.utils import create_input_image
    from tsppp.models.cprm import CriticalPRMModel

    occupancy_map, destinations = instance
    image = create_input_image(occupancy_map, destinations)
    image = torch.tensor(image).unsqueeze(0)

    unet = CriticalPRMModel()
    cmap = unet(image)
    assert cmap.shape == (1, 128, 128)


def test_cprm_pipeline(instance):
    import torch
    from tsppp.data.utils import create_input_image
    from tsppp.models.cprm import CriticalPRMModel
    from tsppp.pipelines.cprm_pipeline import CPRMPipeline

    occupancy_map, destinations = instance

    class UNetTest(CriticalPRMModel):
        def forward(self, images):
            return torch.zeros((1, 128, 128))

    pipeline = CPRMPipeline(unet=UNetTest())
    image = create_input_image(
        occupancy_map, destinations, apply_distance_transform=True
    )
    destinations = torch.from_numpy(destinations.T).unsqueeze(0)
    image = torch.from_numpy(image).unsqueeze(0)
    samples = pipeline(
        destinations,
        image,
        num_samples=100,
    ).samples
    assert samples.shape == (1, 2, 100)


def test_cprm_graph(instance):
    from tsppp.pipelines.ddpm1d_pipeline import PipeplineOutput
    from tsppp.planners.graph_algos import construct_cprm_graph

    occupancy_map, destinations = instance

    class TestPipeline:
        def __call__(
            self,
            destinations,
            images,
            num_samples,
        ):
            return PipeplineOutput(
                np.random.rand(1, 2, num_samples), destinations, images
            )

    pipeline = TestPipeline()
    graph = construct_cprm_graph(
        occupancy_map,
        destinations,
        num_samples=100,
        pipeline=pipeline,
        lambda_value=10.0,
        seed=0,
    )
    assert len(graph.G.nodes) <= len(graph.node_tree.data) > 0


def test_cprm_trainer():
    from tsppp.data.dataset import CPRMDataset
    from tsppp.models.cprm import CriticalPRMModel
    from tsppp.trainers.cprm_trainer import CPRMTrainer

    unet = CriticalPRMModel()
    trainer = CPRMTrainer(
        train_batch_size=2,
        eval_batch_size=2,
        num_epochs=1,
        lr_warmup_steps=1,
        save_sample_epochs=1,
        save_model_epochs=1,
    )
    dataset = CPRMDataset(
        data_file="tests/test_dataset.db",
        apply_distance_transform=True,
    )
    trainer.fit(unet, dataset)
