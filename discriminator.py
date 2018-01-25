'''
The discriminator architecture
Since we are not using Batch Normalization for discriminator set Bias = True
'''
class residual_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=True)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=True)
        self.leaky_relu2 = nn.LeakyReLU()
   
    def forward(self, x):
        o = self.conv1(x)
        o = self.leaky_relu1(o)
        o = self.conv2(o)
        o = o+x
        o = self.leaky_relu2(o)
        return o

class down_sample_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(down_sample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=True)
        self.leaky_relu1 = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        return x

class repeated_block(nn.Module)
    def __init__(self, in_chanel, out_chanel, down_sample_kernel, dropRate=0.0):
        super(repeated_block, self).__init__()
        self.d = down_sample_block(in_chanel,down_sample_kernel,out_chanel,2)
        self.r1 = residual_block(out_chanel,3,out_chanel,1)
        self.r2 = residual_block(out_chanel,3,out_chanel,1)
    
    def forward(self, x):
        x = self.d(x)
        x = self.r1(x)
        x = self.r2(x)
        return x

class discriminator(nn.Module):
	def __init__(self, tags=19):
        super(discriminator, self).__init__()
        self.r1 = repeated_block(3,32,4)
        self.r2 = repeated_block(32,64,4)
        self.r3 = repeated_block(64,128,4)
        self.r4 = repeated_block(128,256,3)
        self.r5 = repeated_block(256,512,3)
        self.d1 = down_sample_block(512,3,1024,2)
        self.dense_1 = nn.Linear(2*2*1024, 1)
        self.dense_tags = nn.Linear(2*2*1024, tags)
        
    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.d1(x)
        x = x.view(x.size()[0],-1)
        return self.dense_1(x),self.dense_tags(x)