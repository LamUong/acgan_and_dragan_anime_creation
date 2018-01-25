'''
The generator architecture
Since we are  using Batch Normalization for discriminator set Bias = False for prev layer
'''
class residual_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chanel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chanel)
   
    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = o+x
        return y

class subpixel_block(nn.Module):
    def __init__(self, in_chanel, kernel_size, out_chanel, stride, dropRate=0.0):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.pixel_shuffler = nn.PixelShuffle(2) 
        self.bn1 = nn.BatchNorm2d(in_chanel)
        self.relu1 = nn.ReLU()
   
    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffler(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x