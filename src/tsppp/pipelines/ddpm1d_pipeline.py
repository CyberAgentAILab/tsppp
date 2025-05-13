"""
DDPMPipeline for 1d data + image condition + reward guidance based on distance-transformed images

Reference code at:
https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/pipelines/ddpm/pipeline_ddpm.py
"""

import warnings
from logging import getLogger
from typing import Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .guidance import get_guidance
from .utils import PipeplineOutput, get_trimmed_path

logger = getLogger(__name__)


class DDPM1DPipeline(DiffusionPipeline):
    """
    DDPM pipeline to generate plausible paths for destinations and images created from TSPPP instances
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        destinations: torch.Tensor,
        images: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 100,
        return_dict: bool = True,
        num_samples: int = None,
        is_trimmed: bool = False,
        guidance_type: str = "dt_10",  # see guidance.py
        **kwargs,
    ) -> Union[PipeplineOutput, Tuple]:
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        num_samples = destinations.shape[0] if num_samples is None else num_samples
        samples = torch.randn(
            (
                num_samples,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
            ),
            generator=generator,
        )
        samples = samples.to(self.device)

        images = images.to(self.device)
        destinations = destinations.to(self.device)

        if guidance_type is not None:
            guidance = get_guidance(guidance_type, images[0, 0], destinations[0])

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # get feature maps if unet has image_encoder
        if hasattr(self.unet, "image_encoder"):
            feature_maps = self.unet.image_encoder.encode(images)
            if feature_maps.shape[0] == 1:
                # logger.info("Repeating feature maps for all samples")
                feature_maps = torch.repeat_interleave(feature_maps, num_samples, dim=0)
        else:
            feature_maps = None

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            model_output = self.unet(
                samples,
                t,
                images=images,
                feature_maps=feature_maps,
            ).sample
            # 2. compute previous image: x_t -> x_t-1
            samples = self.scheduler.step(
                model_output, t, samples, generator=generator
            ).prev_sample
            if guidance_type is not None:
                samples = guidance(samples)

        samples = samples.cpu().numpy()
        destinations = destinations.cpu().numpy()
        images = images.cpu().numpy()

        if is_trimmed:
            samples = get_trimmed_path(samples)

        if not return_dict:
            return (samples, destinations, images)

        return PipeplineOutput(
            samples=samples, destinations=destinations, images=images
        )
