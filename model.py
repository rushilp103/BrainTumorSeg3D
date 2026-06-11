import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

SPATIAL_DIMS = 3
IN_CHANNELS = 4
OUT_CHANNELS = 3
CHANNELS = (16, 32, 64, 128, 256)
STRIDES = (2, 2, 2, 2)


def build_model() -> UNet:
    return UNet(
        spatial_dims=SPATIAL_DIMS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
        strides=STRIDES,
    )


def build_loss() -> DiceCELoss:
    return DiceCELoss(to_onehot_y=False, softmax=True)


if __name__ == "__main__":
    model = build_model()
    model.eval()

    dummy = torch.randn(2, IN_CHANNELS, 128, 128, 128)

    with torch.no_grad():
        output = model(dummy)

    print(f"output shape: {tuple(output.shape)}")

    expected_shape = (2, OUT_CHANNELS, 128, 128, 128)
    assert tuple(output.shape) == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    )
    print("Output shape verified.")
