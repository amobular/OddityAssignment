import pytest
import torch
import torch.nn as nn

from src.models.inflate import InflatedModel


def example_image_and_boring_video():
    image = torch.randn((1, 3, 32, 32))
    video = image.unsqueeze(-3).tile(1, 1, 64, 1, 1)
    return image, video


@torch.no_grad()
def test_conv2d():
    image, video = example_image_and_boring_video()

    image_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0, bias=False),
    )
    video_model = InflatedModel(image_model)

    image_model = image_model.eval()
    video_model = video_model.eval()

    image_features = image_model(image)
    video_features = video_model(video)
    assert torch.allclose(image_features, video_features[:, :, 32], atol=1e-5)


@torch.no_grad()
def test_max_pool2d():
    image, video = example_image_and_boring_video()

    image_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0, bias=False),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    video_model = InflatedModel(image_model)
    image_model = image_model.eval()
    video_model = video_model.eval()

    image_features = image_model(image)
    video_features = video_model(video)
    assert torch.allclose(image_features, video_features[:, :, 16], atol=1e-5)


@torch.no_grad()
def test_avg_pool2d():
    image, video = example_image_and_boring_video()

    image_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    video_model = InflatedModel(image_model)
    image_model = image_model.eval()
    video_model = video_model.eval()

    image_features = image_model(image)
    video_features = video_model(video)
    assert torch.allclose(image_features, video_features[:, :, 16], atol=1e-5)


@torch.no_grad()
def test_adaptive_avg_pool2d():
    image, video = example_image_and_boring_video()

    image_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0, bias=False),
        nn.AdaptiveAvgPool2d(1)
    )
    video_model = InflatedModel(image_model)
    image_model = image_model.eval()
    video_model = video_model.eval()

    image_features = image_model(image)
    video_features = video_model(video)
    assert torch.allclose(image_features, video_features[:, :, 0], atol=1e-5)
