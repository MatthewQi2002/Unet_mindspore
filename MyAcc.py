import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.train as train
import mindspore.ops as ops
from mindspore import nn, Tensor
import os
import cv2

class PixelAcc(train.Metric):
    def __init__(self, num_class=21):
        super(PixelAcc, self).__init__()
        self.clear()
        self._samples_num = 0
        self._acc = 0

    def accuracy(self,y_pred, y):
        tp = np.sum(abs(y_pred.flatten() - y.flatten() < 1e-3), dtype=y_pred.dtype)
        total = len(y_pred.flatten())
        single_acc = float(tp) / float(total)
        return single_acc

    def clear(self):
        self._acc = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = Tensor(inputs[0]).asnumpy()
        y_pred[y_pred > 0.5] = int(1)
        y_pred[y_pred <= 0.5] = int(0)

        y = Tensor(inputs[1]).asnumpy() 
        self._samples_num += y.shape[0]

        for i in range(y.shape[0]):
            single_acc = self.accuracy(y_pred[i], y[i])
            self._acc += single_acc

    def eval(self):
        return self._acc / self._samples_num