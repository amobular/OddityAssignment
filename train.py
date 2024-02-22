from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

CIFAR10_MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
CIFAR10_STD = std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

if __name__ == "__main__":
    # Create datasets with augmentations/transforms
    train_dataset = CIFAR10(root="data", download=True, train=True,
                            transform=transforms.Compose([
                                transforms.AugMix(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                            ]))

    val_dataset = CIFAR10(root="data", download=True, train=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                          ]))
    # Split Training and Validation
    train_idx, val_idx = train_test_split(
        list(range(len(train_dataset))),
        train_size=0.8,
        test_size=0.2,
        random_state=2024,
        stratify=None  # TODO Possibly do this but getting the labels with the CIFAR10 dataset takes a long time
    )

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)
    print(f"Training data length: {len(train_dataset)}, Validation data length: {len(val_dataset)}")

    # Setup dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4,
                                  persistent_workers=True,
                                  drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                                pin_memory=True,
                                num_workers=4,
                                persistent_workers=True,
                                drop_last=True)

    # Setup training

