# *Oddity.ai* - Technical Assignment

A GitHub repository hosting a technical assignment done by Adjorn van Engelenhoven (amobular).

## Installation

I only used two basic libraries: `torch` and `lightning`, which can be done like below (this being the newest CUDA version)
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install lightning
```

Or if you would rather install using a `requirements.txt`: 
```shell
pip3 install -r requirements.txt
```

## Structure
The code has not yet been written but the structure will look something like:
```
checkpoints/                Checkpoint files for models (would normally store these in W&B)
configs/                    Config files for models (would normally store these in W&B)
data/                       Default location for the CIFAR10 dataset
src/                        Source code for models, inflation, etc.
    models/                 Model definition and code for weight inflation
    utils/                  Utility functions like MixUp
tests/                      
train.py                    Training script for the basic CNN               
classify.py                 Classification script for videos
```
