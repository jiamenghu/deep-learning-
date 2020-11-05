import torch 
from torch import nn
import numpy as np

class ResiBlock_v2(nn.Module):
    '''
    预激活残差块；  
    通道 ：channel=64；  
    是否使用BN层 ： BN=True；  
    '''
    def __init__(self,channel=64,BN=True):
        super(ResiBlock_v2,self).__init__()
        self.base=nn.Sequential()
        if BN :
            self.base.add_module('bn_0',nn.BatchNorm2d(channel))
        self.base.add_module('relu_0',nn.ReLU())
        self.base.add_module('conv_0',nn.Conv2d(channel,channel,3,padding=1))
        if BN :
            self.base.add_module('bn_1',nn.BatchNorm2d(channel))
        self.base.add_module('relu_1',nn.ReLU())
        self.base.add_module('conv_1',nn.Conv2d(channel,channel,3,padding=1))
    
        self.init_weight()

    def forward(self,x):
        y = self.base(x)
        return y+x
    def init_weight(self):
        for name,param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                nn.init.kaiming_normal_(param.data, a=0, mode='fan_in', nonlinearity='relu')
            if ('conv' in name) and ('bias' in name):
                param.data.fill_(0.0)

class MultiLayer_ResiBlock_v2(nn.Module):
    '''
    多层V2形式的残差块；  
    通道：channel=64；   
    块的数目：num=3；   
    是否使用BN层：BN=True； 
    '''
    def __init__(self,channel=64,num=3,BN=True):
        super(MultiLayer_ResiBlock_v2,self).__init__()
        self.base=nn.Sequential()
        for i in range(num):
            self.base.add_module('ResiBlock_V2_{}'.format(i),ResiBlock_v2(channel=channel,BN=BN))
        
        self.init_weight()

    def forward(self,x):
        y = self.base(x)
        return y
    def init_weight(self):
        for name,param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                nn.init.kaiming_normal_(param.data, a=0, mode='fan_in', nonlinearity='relu')
            if ('conv' in name) and ('bias' in name):
                param.data.fill_(0.0)

class MultiLayer_ConvLayer(nn.Module):
    '''
    多层卷积层；  
    通道 ：channel=64；   
    层数 ：num=3；  
    是否使用BN层 ： BN=True；  
    '''
    def __init__(self,channel=64,num=3):
        super(MultiLayer_ConvLayer,self).__init__()
        self.base=nn.Sequential()
        for i in range(num):
            self.base.add_module('conv_{}'.format(i),nn.Conv2d(channel,channel,3,padding=1))
            self.base.add_module('bn_{}'.format(i),nn.BatchNorm2d(channel))
            self.base.add_module('relu_{}'.format(i),nn.ReLU())

        self.init_weight()

    def forward(self,x):
        y = self.base(x)
        return y
    def init_weight(self):
        for name,param in self.named_parameters():
            if ('conv' in name) and ('weight' in name):
                nn.init.kaiming_normal_(param.data, a=0, mode='fan_in', nonlinearity='relu')
            if ('conv' in name) and ('bias' in name):
                param.data.fill_(0.0)
