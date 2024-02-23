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


def test(config):
    # Check whether to use cuda
    accelerator = 'cpu'
    if torch.cuda.is_available() and config['use_cuda']:
        accelerator = 'gpu'

    # Create testings dataset with transforms
    training_config = config['training']
    epochs = training_config['epochs']
    batch_size = training_config['batch_size']
    ckpt_path = config['ckpt_path']

    test_dataset = CIFAR10(root="data", download=True, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                           ]))

    # Setup dataloaders
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 num_workers=2,
                                 drop_last=False)

    # Setup training
    model = ResNet(**config['model'])
    wrapper = LightningImageClassificationWrapper.load_from_checkpoint(ckpt_path,
                                                                       model=model, total_steps=-1)

    logger = False  # Could do W&B logging with `pytorch_lightning.loggers.WandbLogger()`

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=epochs,
        accelerator=accelerator,
        precision="16-mixed",
        logger=logger,
    )
    trainer.test(wrapper, dataloaders=test_dataloader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-ckpt", "--checkpoint",
                        help="The path to the checkpoint file containing the weights of the trained model",
                        default="checkpoints/basic_resnet-epoch=149-val_loss=0.353.ckpt")
    parser.add_argument("-c", "--config",
                        help="The path to the config file to create the model",
                        default="configs/basic_resnet.json")
    parser.add_argument("-cuda", "--use-cuda",
                        help="Whether to use CUDA or not, default is true",
                        action="store_false")
    args = parser.parse_args()
    # Load config file
    path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    with open(path, mode='r') as file:
        config = json.load(file)

    config['config_name'] = path.stem
    config['ckpt_path'] = ckpt_path
    config['use_cuda'] = args.use_cuda
    test(config)

    # Results from the basic_resnet config
    # Test Accuracy: 89.3%
    # Test CrossEntropy: 0.3699
