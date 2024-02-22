# Assignment Notes


## Part 1 - Theoretical (Paper)

### Basic Outline of Choices
* Image Size: 224:224
* Temporal Length: 64 Frames (10 frame space at 25 frames per second) 
* Used an InceptionV1 pretrained on ImageNet for both RGB Video and Optical Flow.
  * Does not MaxPool in the time dimension in the beginning.
* Pretrained *again* on Kinetics-400 for the two benchmark datasets.


### Noteworthy Observations
* Using OpticalFlow in of itself can already be better than using RGB Video input.
* ImageNet pretraining in of itself does not increase performance for I3D too much on either dataset ~5%.
* Kinetics-400 pretraining increases the performance on HMDB-51 a lot (66.4 &rarr; 81.3)

### Augmentations
The augmentations in this paper seem a bit lacking, only the following seem to have been used:
* Random ReCrop
* Horizontal Flipping

### Improvements (To be continued...)
* Not using InceptionNetV1. 
  * MaxPooling is generally deemed worse than doing a stride or patch merging
  * `ReLU` is generally deemed worse than something with a gradient in the negative like `GELU` or `SiLU`.
  * Deeper networks are generally deemed to be better these days (though it can have tradeoffs for Oddity's use case)
  * As a possible solution it could be replaced with a small `EfficientNetV2` model.
* Add `AugMix`, `MixUp`, or `CutMix` and then train for longer.
* The model needs better temporal integration
  * Having a receptive field is not necessarily bad, however something like a 3D `SqueezeExcitation` layer can give the model global understanding of channels throughout the time dimension.


## Part 2 - Technical (CNN and Inflated CNNs)

### CNN Training Approach
* Setup the basic dataset, using the `torch` built in `CIFAR10` dataset object, and also create a validation dataset.
* Use the `AugMix` transformation for augmentations, and check the resulting images.
* Create a custom ResNet model, and keep it small.
* Create a Lightning wrapper to handle training, logging, and checkpointing. Also, do `MixUp` if there is time.
* Train the model and check on the test set!

### Weight Inflation Approach
* Write basic functions that inflate different layers used in CNNs
  * Conv2D &rarr; Conv3D
  * BatchNorm2D &rarr; BatchNorm3D
  * MaxPool2D &rarr; MaxPool3D
  * AvgPool2D &rarr; AvgPool3D
* Write tests to check whether the convolution results are the same (as stated in the paper)
* Create a class which takes a CNN with basic layers and converts it to an inflated version, while allowing for customization.
