"""
Modified version of Diffusers' UNet1DModel to be conditional.
Original code at:
https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/unet_1d.py
"""

from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_1d import UNet1DModel, UNet1DOutput

from .encoder import get_encoder


class UNet1DConditionModel(UNet1DModel, ModelMixin, ConfigMixin):
    """
    Conditional UNet model for 1D data.
    This class extends diffusers' UNet1DModel to have a problem instance encoder.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 256,
        sample_rate: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 16,
        time_embedding_type: str = "fourier",
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = False,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = (
            "DownBlock1DNoSkip",
            "DownBlock1D",
            "AttnDownBlock1D",
        ),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        act_fn: str = None,
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
        image_encoder_type: str = "unet",
    ):
        super().__init__(
            sample_size,
            sample_rate,
            in_channels,
            out_channels,
            extra_in_channels
            + 256,  # 256 is the number of channels from the image encoder
            time_embedding_type,
            flip_sin_to_cos,
            use_timestep_embedding,
            freq_shift,
            down_block_types,
            up_block_types,
            mid_block_type,
            out_block_type,
            block_out_channels,
            act_fn,
            norm_num_groups,
            layers_per_block,
            downsample_each_block,
        )

        self.image_encoder = get_encoder(image_encoder_type, in_channels=4)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        images: torch.FloatTensor,
        return_dict: bool = True,
        feature_maps: Optional[torch.FloatTensor] = None,
    ) -> Union[UNet1DOutput, Tuple]:
        """
        Forward pass of the UNet1DCondition model.

        Args:
            sample (torch.FloatTensor): The input sample.
            timestep (Union[torch.Tensor, float, int]): The timestep information.
            images (torch.FloatTensor): The input images.
            return_dict (bool, optional): Whether to return the output as a dictionary. Defaults to True.
            feature_maps (Optional[torch.FloatTensor], optional): The feature maps. Defaults to None.

        Returns:
            Union[UNet1DOutput, Tuple]: The output of the forward pass.

        Note:
            In sample generation, the feature_maps once generated can be reused in the repeated denoising process.
        """

        ### copied from UNet1DModel ###
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed)
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(
                sample.dtype
            )
            timestep_embed = timestep_embed.broadcast_to(
                (sample.shape[:1] + timestep_embed.shape[1:])
            )
        ###############################

        image_embed = self.image_encoder(sample, images, feature_maps=feature_maps)
        timestep_embed = torch.cat([timestep_embed, image_embed], dim=1)

        ### copied from UNet1DModel ###
        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=timestep_embed
            )
            down_block_res_samples += res_samples

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed)

        # 4. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(
                sample, res_hidden_states_tuple=res_samples, temb=timestep_embed
            )

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)

        if not return_dict:
            return (sample,)
        ###############################

        return UNet1DOutput(sample=sample)
