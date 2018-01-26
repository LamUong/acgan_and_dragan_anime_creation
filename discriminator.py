'''
The discriminator architecture
Since we are not using Batch Normalization for discriminator set Bias = True
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class residual_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(residual_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=True)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.conv_2 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=True)
        self.leaky_relu_2 = nn.LeakyReLU()
   
    def forward(self, x):
        r = x
        o = self.conv_1(x)
        o = self.leaky_relu_1(o)
        o = self.conv_2(o)
        o += r
        o = self.leaky_relu_2(o)
        return o

class down_sample_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(down_sample_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=True)
        self.leaky_relu_1 = nn.LeakyReLU()
    
    def forward(self, x):
        o = self.conv_1(x)
        o = self.leaky_relu_1(o)
        return o

class repeated_block(nn.Module)
    def __init__(self, in_chanel, out_chanel, down_sample_kernel, dropRate=0.0):
        super(repeated_block, self).__init__()
        self.down_sample_1 = down_sample_block(in_chanel,down_sample_kernel,out_chanel,2)
        self.residual_1 = residual_block(out_chanel,3,out_chanel,1)
        self.residual_2 = residual_block(out_chanel,3,out_chanel,1)
    
    def forward(self, x):
        o = self.down_sample_1(x)
        o = self.residual_1(o)
        o = self.residual_2(o)
        return o

class discriminator(nn.Module):
	def __init__(self, tags=34):
        super(discriminator, self).__init__()
        self.r_1 = repeated_block(3,32,4)
        self.r_2 = repeated_block(32,64,4)
        self.r_3 = repeated_block(64,128,4)
        self.r_4 = repeated_block(128,256,3)
        self.r_5 = repeated_block(256,512,3)
        self.d_1 = down_sample_block(512,3,1024,2)
        self.dense_1 = nn.Linear(2*2*1024, 1)
        self.dense_tags = nn.Linear(2*2*1024, tags)
        
    def forward(self, x):
        o = self.r_1(x)
        o = self.r_2(o)
        o = self.r_3(o)
        o = self.r_4(o)
        o = self.r_5(o)
        o = self.d_1(o)
        o = o.view(o.size()[0],-1)
        return self.dense_1(o),self.dense_tags(o)