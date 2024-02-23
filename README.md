# *Oddity.ai* - Technical Assignment

A GitHub repository hosting a technical assignment done by Adjorn van Engelenhoven (amobular).

## Installation

I used the following library: `torch`, `lightning`, `scikit-learn`, `torchmetrics`, and `pytest`.
This can be installed which like below (this being the newest CUDA version)
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install lightning scikit-learn torchmetrics pytest
```

Or if you would rather install using a `requirements.txt`: 
```shell
pip3 install -r requirements.txt
```

## Structure
The general structure of this project is setup as follows:
```
checkpoints/                Checkpoint files for models (would normally store these in W&B).
configs/                    Config files for models (would normally store these in W&B).
data/                       Default location for the CIFAR10 dataset.
src/                        Source code for models, inflation, etc.
    lightning/              Lightning wrapper definintion, main training logic is here.
    models/                 Model definition and code for weight inflation.
tests/                      Contains some small tests to test inflation.
train.py                    Training script for the basic CNN.
test.py                     Test script for the basic CNN.
classify.py                 Classification script for videos.
```

##