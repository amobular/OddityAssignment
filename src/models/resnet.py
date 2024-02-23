from typing import List

import torch
import torch.nn as nn


class ResNetBasic(nn.Module):
    """
    The basic ResNet building block from the ResNet paper: https://arxiv.org/pdf/1512.03385.pdf
    The deeper `bottleneck` building block could also be used, but this should work.

    Args:
        in_channels (int): The number of channels of the input tensor.
        out_channels (int): The number of channels this module outputs.
        stride (int): The stride to use for the first convolutional layer.
        dropout_factor (float): The amount of dropout after the basic ResNetBlock.
        downsample (nn.Module): A module that downsamples the residual connection.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, stride: int = 1, dropout_factor: float = 0.25,
                 downsample: nn.Module = None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.downsample = downsample
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_factor)

    def forward(self, x):
        residual = x
        x = self.block(x)

        # Downsample residual connection if necessary
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        # ReLU activation after residual connection
        x = self.activation(x)
        # Apply dropout
        x = self.dropout(x)
        return x


class ResNet(nn.Module):
    """
    A ResNet model as described in https://arxiv.org/pdf/1512.03385.pdf, but with some extra dropout.

    Args:
        layers (List[int]): The number of layers per block, the length indicating the amount of blocks,
            and the integer indicating the amount of ResNetBasic layers to use.
        strides (List[int]): The strides for the first ResNetBasic layer in each block.
        base_width (int): The base amount of channels to convert the RGB image into.
        expansion (int): The multiplicative factor to increase the number of channels after each block.
        n_classes (int): The number of classes to predict (standard 10)
        dropout_factor (float): The amount of dropout to use throughout the network.

    Examples:
        # Creates a 20-layer ResNet without dropout
        model = ResNet(layers=[3, 3, 3], strides=[1, 2, 2], dropout=0.0)
        # Creates a 56-layer ResNet with a dropout of 0.25
        model = ResNet(layers=[9, 9, 9], strides=[1, 2, 2], dropout=0.25)
    """

    def __init__(self, layers: List[int], strides: List[int], base_width: int = 16,
                 expansion=2, n_classes: int = 10, dropout_factor: float = 0.25):
        super().__init__()
        self.layers = layers
        self.strides = strides

        # Make sure the amount of blocks and strides
        assert len(self.layers) == len(self.strides)

        self.base_width = base_width

        # The first convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_factor)
        )

        self.blocks = []
        for block_id, (layer, stride) in enumerate(zip(layers, strides)):
            blocks_layers = []
            downsample = None
            # If this is the first block then keep the base_width else increase the width by the expansion.
            if block_id == 0:
                input_channels = base_width
                out_channels = base_width
            else:
                input_channels = base_width * int((expansion ** (block_id - 1)))
                out_channels = base_width * int((expansion ** block_id))

            # If this block downsamples
            if stride != 1 or input_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(out_channels)
                )

            # Add the downsampling ResNet layer
            blocks_layers.append(
                ResNetBasic(in_channels=input_channels, out_channels=out_channels,
                            stride=stride, downsample=downsample, dropout_factor=dropout_factor)
            )
            # Add the rest of the layers that maintain the size and width
            for _ in range(1, layer):
                blocks_layers.append(
                    ResNetBasic(in_channels=out_channels, out_channels=out_channels,
                                stride=1, downsample=None, dropout_factor=dropout_factor)
                )
            self.blocks += blocks_layers

        # Create sequential layer out all the blocks
        self.blocks = nn.Sequential(*self.blocks)

        # Create a pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Create head
        self.fc = nn.Sequential(
            nn.Linear(in_features=base_width * (expansion ** (len(layers) - 1)), out_features=n_classes),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize parameters the correct way
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet(layers=[2, 2], strides=[1, 2])
    x_in = torch.randn((1, 3, 32, 32))
    x_out = model(x_in)
    print(x_out.size())
