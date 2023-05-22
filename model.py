import mindspore
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.dataset.vision import Inter
from mindspore.dataset import vision
import cv2

def conv(in_ch, out_ch):
    return nn.SequentialCell( 
                              nn.Conv2d(in_ch, out_ch, 3, pad_mode='same'),
                              #nn.BatchNorm2d(out_ch),
                              nn.ReLU(),
                              nn.Conv2d(out_ch, out_ch, 3,pad_mode='same'),
                              #nn.BatchNorm2d(out_ch),
                              nn.ReLU(),
                              )

def upconv (in_ch,out_ch):
    return nn.SequentialCell(
                                nn.Conv2d(in_ch, out_ch, 2, pad_mode='same'),
                                #nn.BatchNorm2d(out_ch),
                                nn.ReLU(),
                              )

def sigmoid_conv(in_ch, out_ch):
    return nn.SequentialCell( 
                              nn.Conv2d(in_ch, out_ch, 1, pad_mode='same'),
                              # nn.BatchNorm2d(out_ch),
                              nn.Sigmoid()
                                 )

class UNet(nn.Cell):
    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.downsample1 = conv(1, 64)
        self.downsample2 = conv(64,128)
        self.downsample3 = conv(128,256)
        self.downsample4 = conv(256,512)
        self.downsample5 = conv(512,1024)
        

        self.upconv1 = upconv(1024,512)
        self.upconv2 = upconv(512,256)
        self.upconv3 = upconv(256,128)
        self.upconv4 = upconv(128,64)

        self.upsample1 = conv(1024,512)
        self.upsample2 = conv(512,256)
        self.upsample3 = conv(256,128)
        self.upsample4 = conv(128,64)

        self.outconv = nn.Conv2d(64,2,3, pad_mode='same')
        self.concat = ops.Concat(axis=1)
        self.sigmoid = ops.Sigmoid()
        self.sigmoid_conv = sigmoid_conv(2,1)
        self.softmax2d = nn.Softmax2d()
        #self.trans_conv1 = nn.Conv2dTranspose(1024, 512,kernel_size=2,stride=2)
        ##self.trans_conv2 = nn.Conv2dTranspose(512,256,kernel_size=2,stride=2)
        #self.trans_conv3 = nn.Conv2dTranspose(256,128,kernel_size=2,stride=2)
        #self.trans_conv4 = nn.Conv2dTranspose(128,64,kernel_size=2,stride=2)
         
    def construct(self, x):
        downfeature1 = self.maxpool(self.downsample1(x))
        downfeature2 = self.maxpool(self.downsample2(downfeature1))
        downfeature3 = self.maxpool(self.downsample3(downfeature2))
        downfeature4 = self.maxpool(self.downsample4(downfeature3))
        downfeature5 = self.downsample5(downfeature4)
        downfeature5  = nn.Dropout2d(p=0.2)(downfeature5)
        
        temp = ops.interpolate(downfeature5,(64,64),mode="nearest")
        temp = self.upconv1(temp)
        # temp = self.trans_conv1(downfeature5)
        up1 = self.concat((self.downsample4(downfeature3),temp))
        upfeature1 = self.upsample1(up1)

        temp = ops.interpolate(upfeature1,(128,128),mode="nearest")    
        temp = self.upconv2(temp) 
        # temp = self.trans_conv2(upfeature1)      
        up2 = self.concat((self.downsample3(downfeature2),temp))
        upfeature2 = self.upsample2(up2)

        temp = ops.interpolate(upfeature2,(256,256),mode="nearest")    
        temp = self.upconv3(temp) 
        # temp = self.trans_conv3(upfeature2)
        up3 = self.concat((self.downsample2(downfeature1),temp))
        upfeature3 = self.upsample3(up3)

        temp = ops.interpolate(upfeature3,(512,512),mode="nearest")    
        temp = self.upconv4(temp) 
        #temp = self.trans_conv4(upfeature3)
        up4 = self.concat((self.downsample1(x),temp))
        upfeature4 = self.upsample4(up4)

        logits = self.outconv(upfeature4)
        output = self.sigmoid_conv(logits)
        
        return output

# model = UNet()

# a = mindspore.Tensor(np.ones((1,1,512,512)),mindspore.float32)
# print(a)
# print(model.construct(a).shape)
# print(a)

# print(np.array(model.construct(a)))
