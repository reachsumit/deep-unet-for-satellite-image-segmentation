# Deep UNet for satellite image segmentation

![banner](https://i.imgur.com/hjITfpc.png)!

## About this project
This is a Keras based implementation of a deep UNet that performs satellite image segmentation.

## Dataset
* The dataset consists of 8-band commercial grade satellite imagery taken from SpaceNet dataset.
* Train collection contains few tiff files for each of the 24 locations.
* Every location has an 8-channel image containing spectral information of several wavelength channels (red, red edge, coastal, blue, green, yellow, near-IR1 and near-IR2). These files are located in data/mband/ directory.
* Also available are correctly segmented images of each training location, called mask. These files contain information about 5 different classes: buildings, roads, trees, crops and water (note that original Kaggle contest had 10 classes).
* Resolution for satellite images s 16-bit. However, mask-files are 8-bit.

## Implementation
* Deep Unet architecture is employed to perform segmentation.
* Image augmentation is used for input images to significantly increases train data.
* Image augmentation is also done while testing, mean results are exported to result.tif image.
![examples](https://i.imgur.com/34lq5bD.jpg)

**Note:** Training for this model was done on a Tesla P100-PCIE-16GB GPU.

## Prediction Example
![prediction example](https://i.imgur.com/CalIxTU.png)

## Network architecture
![Deep Unet Architecture](https://i.imgur.com/zX1r5Rx.png)
