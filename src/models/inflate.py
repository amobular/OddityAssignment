from typing import Tuple
import torch
import torch.nn as nn
import torchvision


def expand_tuple(t: Tuple) -> Tuple:
    return min(t), *t


def inflate_argument(t: Tuple | int):
    if isinstance(t, int):
        return t
    if isinstance(t, tuple):
        return expand_tuple(t)


def inflate_convolution(layer: nn.Conv2d):
    has_bias = layer.bias is not None
    replacement_kernel_size = expand_tuple(layer.kernel_size)
    replacement_padding = expand_tuple(layer.padding) if isinstance(layer.padding, tuple) else layer.padding

    # Create a replacement convolution
    replacement = nn.Conv3d(in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            bias=has_bias,
                            kernel_size=replacement_kernel_size,
                            stride=inflate_argument(layer.stride),
                            dilation=inflate_argument(layer.dilation),
                            padding=replacement_padding)

    # Add time-wise dimension:
    #   (out_channels, in_channels, k_height, k_width) -> (out_channels, in_channels, 1, k_height, k_width)
    inflated_weight = layer.weight.unsqueeze(-3)
    # Add inflation of time-wise dimension
    #   (out_channels, in_channels, 1, k_height, k_width) -> (out_channels, in_channels, k_time, k_height, k_width)
    inflated_weight = inflated_weight.tile(1, 1, replacement_kernel_size[0], 1, 1) / replacement_kernel_size[0]
    replacement.weight = nn.Parameter(inflated_weight)
    if has_bias:
        replacement.bias = nn.Parameter(layer.bias)
    return replacement


def inflate_batch_norm(layer: nn.BatchNorm2d):
    replacement = nn.BatchNorm3d(
        layer.num_features
    )

    replacement.weight = layer.weight
    replacement.bias = layer.bias
    return replacement


def inflate_max_pooling(layer: nn.MaxPool2d):
    replacement = nn.MaxPool3d(
        kernel_size=inflate_argument(layer.kernel_size),
        stride=inflate_argument(layer.stride),
        dilation=inflate_argument(layer.dilation),
        padding=inflate_argument(layer.padding),
    )
    return replacement


def inflate_avg_pooling(layer: nn.AvgPool2d):
    replacement = nn.AvgPool3d(
        kernel_size=inflate_argument(layer.kernel_size),
        stride=inflate_argument(layer.stride),
        padding=inflate_argument(layer.padding),
    )
    return replacement


def inflate_adaptive_avg_pooling(layer: nn.AdaptiveAvgPool2d):
    replacement = nn.AdaptiveAvgPool3d(
        output_size=inflate_argument(layer.output_size)
    )
    return replacement


class InflatedModel(nn.Module):
    """
    General purposed module wrapper to inflate a 2D CNN according to https://arxiv.org/pdf/1705.07750.pdf
    Will generally work as long as the internal model does not make use of the functional library.

    Args:
        model (nn.Module): A trained model that will be inflated.

    Examples:
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        inflated_model = InflatedModel(model)
        img = torch.randn((1, 3, 64, 256, 256))
        output = inflated_model(img)
    """

    INFLATABLE_MODULES = [
        nn.BatchNorm2d,
        nn.Conv2d,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
    ]

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model = self.create_inflated_replacement(self.model)

    def create_inflated_replacement(self, module: nn.Module):
        """
        Recursively replaces a modules children with their inflated counterparts.
        If the module has any children, i.e. a module of type Sequential, ModuleList, ModuleDict or a custom Module,
        then replace their children with their inflated counterparts.
        If the module does not have children it is a basic building block there are two cases:
            1. The module is inflatable (like Conv2D), we return an inflated replacement
            2. The module is not inflatable (like Linear, or ReLU), we return the original.

        :param module: The module to inflate
        :type module: nn.Module
        :return: The inflated module
        :rtype: nn.Module
        """
        # If the Module object contains other modules
        if len(list(module.children())) > 0:
            # For every child module create an inflated replacement and replace the current child
            for name, child in module.named_children():
                replacement = self.create_inflated_replacement(child)
                module.__setattr__(name, replacement)
            return module

        # If the module is inflatable return the inflated module, else simply return the original layer
        if any(isinstance(module, inflatable_module) for inflatable_module in self.INFLATABLE_MODULES):
            return self.inflate_module(module)
        else:
            return module

    @staticmethod
    def inflate_module(module):
        """
        Checks the type of module and applies the appropriate inflation method
        :param module: The module that needs to be inflated
        :type module: One of INFLATABLE_MODULES
        :return: The inflated module.
        :rtype: The same type as a module.
        """
        if isinstance(module, nn.Conv2d):
            return inflate_convolution(module)
        if isinstance(module, nn.BatchNorm2d):
            return inflate_batch_norm(module)
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return inflate_adaptive_avg_pooling(module)
        if isinstance(module, nn.MaxPool2d):
            return inflate_max_pooling(module)
        if isinstance(module, nn.AvgPool2d):
            return inflate_avg_pooling(module)
        raise ValueError(f"Cannot inflate module of type: {type(module)}...")

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # A little test of the example
    model = torchvision.models.resnet18(weights=None)
    inflated_model = InflatedModel(model)
    img = torch.randn((1, 3, 64, 256, 256))
    output = inflated_model(img)
    print(output.size())
