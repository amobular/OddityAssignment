import json
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Subset, DataLoader

from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from src.lightning.wrapper import LightningImageClassificationWrapper
from src.models.resnet import ResNet

CIFAR10_MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
CIFAR10_STD = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]


def train(config):
    # Check whether to use cuda
    accelerator = 'cpu'
    if torch.cuda.is_available() and config['use_cuda']:
        accelerator = 'gpu'

    # Create datasets with augmentations/transforms
    training_config = config['training']
    epochs = training_config['epochs']
    batch_size = training_config['batch_size']
    seed = training_config['seed']

    train_dataset = CIFAR10(root="data", download=True, train=True,
                            transform=transforms.Compose([
                                transforms.AugMix(severity=3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                            ]))

    val_dataset = CIFAR10(root="data", download=True, train=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                          ]))
    # Split Training and Validation (90/10)
    pl.seed_everything(seed=seed)
    train_idx, val_idx = train_test_split(
        list(range(len(train_dataset))),
        train_size=0.9,
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=None  # TODO Possibly stratify but getting the labels with the CIFAR10 object takes a long time
    )

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)
    print(f"Training data length: {len(train_dataset)}, Validation data length: {len(val_dataset)}")

    # Setup dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=6,
                                  persistent_workers=True,
                                  drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                num_workers=4,
                                persistent_workers=True,
                                drop_last=True)

    # Setup training
    model = ResNet(**config['model'])
    wrapper = LightningImageClassificationWrapper(model, total_steps=epochs * len(train_dataloader))

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{config['config_name']}" + "-{epoch}-{val_loss:.3f}",
    )

    logger = False  # Could do W&B logging with `pytorch_lightning.loggers.WandbLogger()`

    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        deterministic=True,
        max_epochs=epochs,
        accelerator=accelerator,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="The path to the config file to use when training",
                        default="configs/basic_resnet.json")
    parser.add_argument("-cuda", "--use-cuda",
                        help="Whether to use CUDA or not, default is true",
                        action="store_false")
    args = parser.parse_args()
    # Load config file
    path = Path(args.config)
    with open(path, mode='r', encoding='utf-8') as file:
        config = json.load(file)
    config['config_name'] = path.stem
    config['use_cuda'] = args.use_cuda
    train(config)
