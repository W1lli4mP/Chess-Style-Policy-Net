import torch
import torch.nn as nn

KERNEL_SIZE = 3
PADDING = 1
OUT_CHANNELS = 64 # number of filters/feature maps

#* inherit pytorch's nn and extend it
#? why? because it might be useful idk
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        #* 1 res block has 2 convolution layers with 2 batch norms
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=KERNEL_SIZE,
            padding=PADDING
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=KERNEL_SIZE,
            padding=PADDING
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.tensor):
        # y = F(x) + x
        residual = x

        fx = self.conv1(x)
        fx = self.bn1(y)
        fx = self.relu(y)

        fx = self.conv2(y)
        fx = self.bn2(y)
        
        y = fx + residual
        y = self.relu(y)

        return y