import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def get_encoder(encoder_type: str, **kwargs) -> nn.Module:
    if encoder_type == "unet":
        return UNetEncoder(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")


@torch.jit.script
def _extract_features(X, Y):
    """
    Extracts features from the encoded images along given sequences.

    Args:
        X (torch.Tensor): Encoded feature maps.
        Y (torch.Tensor): Input sequences (i.e., path).

    Returns:
        torch.Tensor: Extracted features.
    """
    map_size = X.shape[-1]
    Y = ((Y + 1) * map_size / 2).long()
    Y = torch.clamp(Y, 0, map_size - 1)

    features = []
    for X_, Y_ in zip(X, Y):
        features.append(X_[:, Y_[0], Y_[1]])

    return torch.stack(features)


class UNetEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        encode_dim: int = 256,
        in_channels: int = 1,
        encoder_name: str = "vgg16_bn",
    ):
        DECODER_CHANNELS = [256, 128, 64, 32, 16]
        super().__init__()
        self.num_layers = num_layers
        self.encode_dim = encode_dim
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=encode_dim,
            in_channels=in_channels,
            encoder_depth=num_layers,
            decoder_channels=DECODER_CHANNELS[:num_layers],
        )
        self.add_module("net", self.net)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input images created from TSPPP instances

        Args:
            images (torch.Tensor): Input images to be encoded.

        Returns:
            torch.Tensor: Encoded feature maps.
        """
        feature_maps = self.net(images)
        return feature_maps

    def forward(
        self,
        sequences: torch.Tensor,
        images: torch.Tensor,
        feature_maps: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the encoder model.

        Args:
            sequences (torch.Tensor): Tensor of shape (batch_size, 2, seq_length) representing the input sequences.
            images (torch.Tensor): Tensor representing the input images.
            feature_maps (torch.Tensor, optional): Tensor representing the feature maps. Defaults to None.

        Returns:
            torch.Tensor: Tensor representing the extracted features.

        Note:
            It skips the encoding step if the feature maps are provided.
        """

        assert (
            sequences.shape[1] == 2
        ), "Sequences must be of shape (batch_size, 2, seq_length)"
        assert images.shape[-2] == images.shape[-1], "Images must be square"

        if feature_maps is None:
            feature_maps = self.encode(images)
        features = _extract_features(feature_maps, sequences)
        return features

    def get_encode_dim(self):
        """
        Get the dimension of the encoded features.

        Returns:
            int: Dimension of the encoded features.
        """
        return self.encode_dim
