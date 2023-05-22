import os
import cv2
from mindspore.dataset import vision, GeneratorDataset
from mindspore.dataset.vision import Inter
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
import glob
import matplotlib.pyplot as plt

class ISBI_Train_Data:
    def __init__(self,data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.label_path = glob.glob(os.path.join(data_path, 'label/*.png'))

    def __getitem__(self, index):
        
        image = cv2.imread(self.imgs_path[index], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)
        image = image.reshape((image.shape[0], image.shape[1], 1))
        label = label.reshape((label.shape[0], label.shape[1], 1))
        
        return image, label

    def __len__(self):
        return len(self._data)
    
    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        return len(self.imgs_path)
    




def load_train_data(data_path,img_size,lable_size,batch_size,shuffle):
    image_transforms = [
        # vision.Resize(img_size, interpolation=Inter.BILINEAR),
        vision.RandomRotation(degrees=2.0),
        # vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=0.4),
        vision.Rescale(1.0 / 255.0, 0),
        vision.HWC2CHW()
    ]

    dataset = ISBI_Train_Data(data_path)
    dataset = GeneratorDataset(dataset, dataset.column_names,shuffle=shuffle)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(image_transforms, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

