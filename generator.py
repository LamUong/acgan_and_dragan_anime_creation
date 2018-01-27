'''
The generator architecture
Since we are  using Batch Normalization for discriminator set Bias = False for prev layer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class residual_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(residual_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_chanel)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_chanel)
   
    def forward(self, x):
        r = x
        o = self.conv_1(x)
        o = self.bn_1(o)
        o = self.relu_1(o)
        o = self.conv_2(o)
        o = self.bn_2(o)
        o += r
        return o

class subpixel_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(subpixel_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.pixel_shuffler = nn.PixelShuffle(2) 
        self.bn_1 = nn.BatchNorm2d(in_chanel)
        self.relu_1 = nn.ReLU()
   
    def forward(self, x):
        o = self.conv_1(x)
        o = self.pixel_shuffler(o)
        o = self.bn_1(o)
        o = self.relu_1(o)
        return o

class generator(nn.Module):
    def __init__(self, tags=34):
        super(generator, self).__init__()
        self.dense_1 = nn.Linear(128+tags, 16*16*64)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()
        self.residual = self.get_residual_blocks(16)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()
        self.subpixel = self.get_subpixel_blocks(3)
        self.conv_1 = nn.Conv2d(64, 3, kernel_size=9, stride=1,padding=4, bias=True)
        self.tanh_1 = nn.Tanh()


    def get_residual_blocks(self,num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(residual_block(64,3,64,1))
        return nn.Sequential(*layers)

    def get_subpixel_blocks(self,num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(subpixel_block(64,3,256,1))
        return nn.Sequential(*layers)      
        
    def forward(self, x):
        o = self.dense_1(x)
        o = o.view(-1,64,16,16)
        o = self.bn_1(o)
        o = self.relu_1(o)
        r = o
        o = self.residual(o)
        o = self.bn_2(o)
        o = self.relu_2(o)
        o += r       
        o = self.subpixel(o)
        o = self.conv_1(o)
        o = self.tanh_1(o)
        return o
'''
g = generator()
x = Variable(torch.ones(10, 128+34), requires_grad=True)
print(g(x))
'''