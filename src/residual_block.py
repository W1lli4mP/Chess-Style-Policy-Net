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

        #* 1 res block has 2 convolution layers
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=PADDING
        )

        self.conv2 = nn.Conv2d(
            in_channels=OUT_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=PADDING
        )

    # temp
    def get_residual_output(self, x: torch.tensor):
        # x is either the encoded board tensor or y from another res block
        #! resolve input channel mismatch
        # produce feature maps
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        #! could use
        #! nn.Sequential(self.conv1, self.conv2)

        return y2