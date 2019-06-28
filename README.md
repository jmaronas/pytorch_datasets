# PyTorch Datasets

I create some unavailable pytorch datasets that I will try to push into the torchvision package.

## Compatibility

  These datasets are preparared for at least torchvision 0.2.2 and require scipy version 1.2.1. They have been programmed 
  reusing code and following the same structure as other provided datasets. Basically, fucntions that appear in orchvision 0.3.0
  but not in torchvision 0.2.2 has been directly copied into the file so as to be able to easily incorporate in the main project.


* [tinyImageNet](https://tiny-imagenet.herokuapp.com/)
* [caltech birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (a version for classification and a version for object detection)
* [standford cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### tiny ImageNet

An example of how to use tiny ImagetNet is provided in the file check_tiny_imagenet.py

### Birds

This dataset contains images of birds with different shapes. The dataset contain many information that can be used for different task, but the datasets provided are thought for object segmentation and classification. An example of how to use Birds is provided in the file check_birds.py

#### Classification: 

For classification each image is reshaped to have the same size.  The set of operations performed are: 

* 1) Crop the images using the bounding box provided

* 2) Pad the images to have a square size, using a padding specified as argument. This operation is done using numpy.pad, thus one can choose any option available under this method

* 3) Reshape images to user-specified image shape using an interpolation method which is provided as argument.


For efficiency, the dataset is processed once and store in a folder from which after, images are loaded. A checkpoint is introduced just in case something is wrong during this processing step, thus ensuring that it will correctly finished. If there is a problem during this step, you will be asked to erase the folder where your processed data is placed.

With this processing you can get competitive results on this task ~80% using pretrained models on ImageNet.


#### Object detection: 

In this case the code returns the images, the labels and a numpy array containing the bounding boxes.

The dataset contains a method: dataset.default_collate which can be passed directly to a dataloader.


### TODO: 

* documentation for cars
* refactor some code, for instance repeated functions
* clearly state what is neede to be incorporated in the current functions from torchvision 
* attributes from cars class
      


