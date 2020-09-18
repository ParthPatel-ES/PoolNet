


import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F
import torch.nn.quantized.functional as qF

#from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, model_urls
#from torchvision.models.utils import load_state_dict_from_url
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch._jit_internal import Optional
from torchvision.models.quantization.utils import _replace_relu, quantize_model

from typing import List, Optional
from torch.nn.modules.utils import _pair, _triple
from torch import Tensor

#affine_par = True

def conv3x3(in_planes, out_planes, stride=1):

    print(in_planes)
    "3x3 convolution with padding"
    return nnq.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nnq.BatchNorm2d(planes) # ,affine = affine_par
        self.relu = nnq.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nnq.BatchNorm2d(planes) #, affine = affine_par
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nnq.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nnq.BatchNorm2d(planes) #,affine = affine_par

        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4

        self.conv2 = nnq.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nnq.BatchNorm2d(planes) #,affine = affine_par
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nnq.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nnq.BatchNorm2d(planes * 4) #, affine = affine_par
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nnq.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):

        self.inplanes = 64

        super(ResNet, self).__init__()

        self.conv1 = nnq.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nnq.BatchNorm2d(64) #,affine = affine_par

        for i in self.bn1.parameters():
            i.requires_grad = False

        self.relu = nnq.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change 
        #self.maxpool = qF.max_pool2d(x = None ,kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 2)
        self.quant0 = torch.quantization.QuantStub()
        self.quant1 = torch.quantization.QuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.quant3 = torch.quantization.QuantStub()
        
        self.dequant = torch.quantization.DeQuantStub()
        for m in self.modules():
            #print(m) print(type(m))
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #zero_init_residual might improve performance from core development

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nnq.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nnq.BatchNorm2d(planes * block.expansion), #,affine = affine_par
            )

        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):

        tmp_x = []
        x = self.quant0(x)
        x = self.conv1(x)
        x = self.dequant(x)

        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)

        #x = qF.max_pool2d(x,kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)
        
        return tmp_x


class ResNet_locate(nn.Module):
    def __init__(self, block, layers):

        super(ResNet_locate,self).__init__()
        self.resnet = ResNet(block, layers)
        self.in_planes = 512
        self.out_planes = [512, 256, 256, 128]

        self.ppms_pre = nnq.Conv2d(2048, self.in_planes, 1, 1, bias=False)

        ppms, infos = [], []
        for ii in [1, 3, 5]:
           
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nnq.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nnq.ReLU(inplace=False)))
             
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nnq.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nnq.ReLU(inplace=False))

        for ii in self.out_planes:
            infos.append(nn.Sequential(nnq.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nnq.ReLU(inplace=False)))
            

        self.infos = nn.ModuleList(infos)
        #print(self.infos)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

def resnet50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model

#testModel = resnet50_locate()
#testVar = testModel
#print(testVar)
#print(testModel)
