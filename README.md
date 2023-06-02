# U-Net_mindspore
This is an implementation of U-Net based on Mindspore. The project consists of two parts:  
+ Convert hdf5 model to mindspore ckpt. I train the model [unet](https://github.com/zhixuhao/unet) and get an hdf5 format model. Then, I convert it to ckpt.
+ Train the U-Net model using mindspore.  


Paper Link: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
## Overview
+ [model.py](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/model.py) : the U-Net structure
+ [dataloader.py](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/dataloader.py) : to load train data and label
+ [MyAcc.py](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/MyAcc.py) : the calculation of pixel accuracy, which is used to evaluate the result
+ [train.ipynb](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/train.ipynb) : to train and save the model. Pixel accuracy can reach about 93% after training 1000 epochs.
+ [convert.ipynb](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/convert.ipynb) : to convert the hdf5 model to mindspore ckpt model
## Data
The data is from ISBI Challenge.
+ [train-volume.tif](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/train-volume.tif)
+ [train-labels.tif](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/train-labels.tif)
+ [test-volume.tif](https://github.com/MatthewQi2002/Unet_mindspore/blob/main/test-volume.tif)   


I have convert the train dataset to png. You can find it in "train.zip".  
Due to the fact that the label of the test set is currently not publicly available, I use picture 4,9,14,19,24 in the training set to evaluate the trained model.  
## Dependencies
I trained the model on ModelArts Notebook provided by Huawei. However, it also works on your own PC once you have installed:  
+ mindspore 2.0.0 nightly
+ GPU Cuda 10.1
+ Python 3.7  

You can get it from [install mindspore](https://mindspore.cn/install).
