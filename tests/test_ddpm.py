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


def test_ddpm_unet(instance):
    import torch
    from tsppp.data.utils import create_input_image
    from tsppp.models.unet_1d_condition import UNet1DConditionModel

    occupancy_map, destinations = instance
    image = create_input_image(occupancy_map, destinations)
    image = torch.tensor(image).unsqueeze(0)

    unet = UNet1DConditionModel()
    feature_map = unet.image_encoder.encode(image)
    sample = unet(
        torch.zeros((1, 2, 256)), torch.zeros(1), images=image, feature_maps=feature_map
    ).sample
    assert sample.shape == (1, 2, 256)


def test_ddpm_pipeline(instance):
    import torch
    from diffusers import DDPMScheduler
    from tsppp.data.utils import create_input_image
    from tsppp.models.unet_1d_condition import UNet1DConditionModel, UNet1DOutput
    from tsppp.pipelines.ddpm1d_pipeline import DDPM1DPipeline

    occupancy_map, destinations = instance

    class UNetTest(UNet1DConditionModel):
        def forward(self, sample, timestep, images, feature_maps):
            return UNet1DOutput(sample=sample)

    scheduler = DDPMScheduler(num_train_timesteps=5)

    pipeline = DDPM1DPipeline(unet=UNetTest(sample_size=256), scheduler=scheduler)
    image = create_input_image(
        occupancy_map, destinations, apply_distance_transform=True
    )
    destinations = torch.from_numpy(destinations.T).unsqueeze(0)
    image = torch.from_numpy(image).unsqueeze(0)
    samples = pipeline(
        destinations,
        image,
        num_inference_steps=5,
        num_samples=5,
        guidance_type=None,
        is_trimmed=False,
    ).samples
    assert samples.shape == (5, 2, 256)


def test_ddpm_graph(instance):
    from tsppp.pipelines.ddpm1d_pipeline import PipeplineOutput
    from tsppp.planners.graph_algos import construct_ddpm_graph

    occupancy_map, destinations = instance

    class TestPipeline:
        def __call__(
            self,
            destinations,
            images,
            num_inference_steps,
            num_samples,
            generator=None,
            is_trimmed=False,
            guidance_type=None,
        ):
            return PipeplineOutput(
                np.random.rand(num_samples, 2, 256), destinations, images
            )

    pipeline = TestPipeline()
    graph = construct_ddpm_graph(
        occupancy_map,
        destinations,
        num_samples=5,
        pipeline=pipeline,
        num_neighbors=5,
        num_inference_steps=5,
        add_boundary_nodes=False,
        seed=0,
    )
    assert len(graph.G.nodes) == len(graph.node_tree.data) > 0


def test_ddpm_trainer():
    from tsppp.data.dataset import DDPMDataset
    from tsppp.models.unet_1d_condition import UNet1DConditionModel
    from tsppp.trainers.ddpm_trainer import DDPMTrainer

    unet = UNet1DConditionModel(sample_size=256)
    trainer = DDPMTrainer(
        train_batch_size=2,
        eval_batch_size=2,
        num_epochs=1,
        num_train_timesteps=10,
        lr_warmup_steps=1,
        save_sample_epochs=1,
        save_model_epochs=1,
    )
    dataset = DDPMDataset(
        data_file="tests/test_dataset.db",
        apply_distance_transform=True,
    )
    trainer.fit(unet, dataset)
