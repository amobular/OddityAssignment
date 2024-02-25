import torch
import numpy as np

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""
Classify a video as an airplane, automobile, bird, cat, deer, dog, frog, horse,
ship or truck.

Args:
    video: NumPy `ndarray` with in format `(N, H, W, C)` and shape `(64, 32, 32, 3)`
           with type `np.float64`, pixel values normalized in the range 0.0 to 1.0.
Returns:
    A dictionary with each of the classes as keys, and the corresponding normalized floating-point scores as values.

Example:
    ```
    import numpy as np
    x = np.zeros((64, 32, 32, 3))
    y = classify(x)
    print(y)
    ```
Example output:

    ```
    {
    'airplane': 0.1,
    'automobile': 0.1,
    'bird': 0.1,
    'cat': 0.1,
    'deer': 0.1,
    'dog': 0.1,
    'frog': 0.1,
    'horse': 0.1,
    'ship': 0.1,
    'truck': 0.1
    }
    ```

"""
@torch.no_grad()
def classify(video: np.ndarray):
    import json
    from collections import OrderedDict
    from src.models.inflate import InflatedModel
    from src.models.resnet import ResNet
    from test import CIFAR10_MEAN, CIFAR10_STD

    # Load model
    config_path = "configs/basic_resnet.json"
    checkpoint_path = "checkpoints/basic_resnet-epoch=149-val_loss=0.353.ckpt"

    config = json.load(open(config_path, mode='r'))

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    # Remove the model wrapper from the state_dict
    state_dict = OrderedDict({".".join(k.split(".")[1:]): v for k, v in state_dict.items()})

    model = ResNet(**config['model'])
    model.load_state_dict(state_dict)
    model = InflatedModel(model)
    model.eval()

    # Prepare video input
    video = (video - np.array(CIFAR10_MEAN)) / np.array(CIFAR10_STD)
    # Create tensor from numpy array and put shapes in correct order: (N, H, W, C) -> (C, N, H, W)
    video = torch.from_numpy(video).to(torch.float32).permute(-1, 0, 1, 2)
    # Add batch dimension
    video = video.unsqueeze(0)

    # Get output logits
    prediction = torch.softmax(model(video), dim=-1)[0].tolist()

    # Create output dict
    prediction = dict(zip(CIFAR10_CLASSES, prediction))

    return prediction


if __name__ == "__main__":
    x = np.zeros((64, 32, 32, 3))
    y = classify(x)
    print(y)
