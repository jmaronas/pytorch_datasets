# PyTorch Datasets

I create some unavailable pytorch datasets that I will try to push into the torchvision package.

## Compatibility

  These datasets are preparared for at least torchvision 0.2.2 and require scipy version 1.2.1 and progress module to be installed. They have been programmed 
  reusing code and following the same structure as other provided datasets. Basically, functions that appear in torchvision 0.3.0
  but not in torchvision 0.2.2 has been directly copied into a file named common.py. With this, incorporating these datasets in the main project is straightforward.

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


For efficiency, the dataset is processed once and store in a folder from which after, images are loaded. A checkpoint is introduced to check if something goes wrong during this processing step, thus ensuring that it will correctly finished. If there is a problem during this step, you will be asked to erase the folder where your processed data is placed.

With this processing you can get competitive results on this task ~80% using pretrained models on ImageNet.


#### Object detection: 

In this case the code returns the images, the labels and a numpy array containing the bounding boxes.

The dataset contains a method: dataset.default_collate which can be passed directly to a dataloader.


### Cars

This dataset contains images of cars with different shapes. An example of how to use Cars is provided in the file check_cars.py

#### Classification: 

For classification each image is reshaped to have the same size.  The set of operations performed are: 

* 1) Crop the images using the bounding box provided

* 2) Pad the images to have a square size, using a padding specified as argument. This operation is done using numpy.pad, thus one can choose any option available under this method

* 3) Reshape images to user-specified image shape using an interpolation method which is provided as argument.


For efficiency, the dataset is processed once and store in a folder from which after, images are loaded. A checkpoint is introduced to check if something goes wrong during this processing step, thus ensuring that it will correctly finished. If there is a problem during this step, you will be asked to erase the folder where your processed data is placed.

With this processing you can get competitive results on this task ~88% using pretrained models on ImageNet.


#### Object detection: 

In this case the code returns the images, the labels and a numpy array containing the bounding boxes.

The dataset contains a method: dataset.default_collate which can be passed directly to a dataloader. See the code provided 




# What needs to be added/changed in the torchvision project?

In order to use this datasets we need a slight modification in the torchvision project version 0.3.0. Function ``` def _is_gzip(filename)  ``` available at torchvision.datasets.utils has to be changed from:

```
def _is_targz(filename):
	return filename.endswith(".tar.gz")

```

to

```
def _is_targz(filename):
        return filename.endswith(".tar.gz") or filename.endswith(".tgz")
```



      


