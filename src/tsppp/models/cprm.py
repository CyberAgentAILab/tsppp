import segmentation_models_pytorch as smp
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class CriticalPRMModel(ModelMixin, ConfigMixin):
    """
    UNet model used to estimate criticality maps for CriticalPRM
    """

    @register_to_config
    def __init__(
        self,
        num_layers: int = 4,
        in_channels: int = 4,
        encoder_name: str = "vgg16_bn",
    ):
        DECODER_CHANNELS = [256, 128, 64, 32, 16]
        super().__init__()
        self.num_layers = num_layers
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=1,
            in_channels=in_channels,
            encoder_depth=num_layers,
            decoder_channels=DECODER_CHANNELS[:num_layers],
        )
        self.add_module("net", self.net)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Estimate criticality maps from TSPPP images.

        Args:
            images (torch.Tensor): Input images to be encoded.

        Returns:
            torch.Tensor: Criticality maps.
        """
        criticality_maps = self.net(images)[:, 0]
        return criticality_maps
