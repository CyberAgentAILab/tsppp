from dataclasses import dataclass
from logging import getLogger

import torch
import torch.nn as nn

from ..models.cprm import CriticalPRMModel
from .utils import PipeplineOutput

logger = getLogger(__name__)


@dataclass
class CPRMPipeline:
    """
    CPRM pipeline to generate plausible paths for destinations and images created from TSPPP instances
    """

    unet: nn.Module

    def save_pretrained(self, savedir: str):
        self.unet.save_pretrained(f"{savedir}/unet")

    @classmethod
    def from_pretrained(self, savedir: str):
        unet = CriticalPRMModel.from_pretrained(f"{savedir}/unet")
        return self(unet)

    @property
    def device(self) -> torch.device:
        return self.unet.device

    @torch.no_grad()
    def __call__(
        self,
        destinations: torch.Tensor,
        images: torch.Tensor,
        num_samples: int,
    ) -> PipeplineOutput:
        criticality_maps = torch.sigmoid(self.unet(images)).float()
        pos = torch.vstack(
            torch.where(torch.ones_like(criticality_maps[0], device=images.device))
        ).T

        sampled_pos = torch.zeros(
            (images.shape[0], 2, num_samples), device=images.device
        )
        for i, c_map in enumerate(criticality_maps):
            prob = c_map.flatten()
            sampled_id = torch.multinomial(prob, num_samples, replacement=True)
            sampled_pos[i] = pos[sampled_id].T
        samples = sampled_pos.cpu().numpy()
        destinations = destinations.cpu().numpy()
        images = images.cpu().numpy()

        return PipeplineOutput(
            samples=samples, images=images, destinations=destinations
        )
