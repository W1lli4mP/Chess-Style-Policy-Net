import torch
import torch.nn as nn

KERNEL_SIZE = 3
PADDING = 1

#* inherit pytorch's nn and extend it
#? implement forward(), making it compatible with Sequential()
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        #* 1 res block has 2 convolution layers with 2 batch norms
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=KERNEL_SIZE,
            padding=PADDING,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=KERNEL_SIZE,
            padding=PADDING,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # y = F(x) + x
        residual = x

        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)

        fx = self.conv2(fx)
        fx = self.bn2(fx)
        
        y = fx + residual
        y = self.relu(y)

        return y