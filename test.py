from discriminator import discriminator
from generator import generator
from data_loader import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable, grad
from torchvision.transforms import ToTensor
at_p='/home/elamuon/paper/tag_detection/illustration2vec/cleaned_image_atrri.p'
image_dir = '/home/elamuon/paper/tag_detection/illustration2vec/images/clean_resized_image/'
lambda_gp = 0.5
batch_size = 64
learning_rate=0.0002
beta_1 = 0.5
noise_input_dim = 128
max_epochs = 200
lambda_adv = 19
tag_list = [
 'blush',
 'smile',
 'glasses',
'long hair', 'short hair', 
'blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair',
'blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes', 'grey eyes']
misc = ['blush','smile','glasses']
hair_type=['long hair', 'short hair'] 
hair_color= ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair']
eyes_color= ['blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes', 'grey eyes']
tag_index = {}
for i in range(len(tag_list)):
    tag_index[tag_list[i]] = i

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, iterations):
    """Sets the learning rate to the initial LR decayed by 10 every 50000 iterations"""
    lr = learning_rate * (0.1 ** (iterations // 50000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)

def get_random_numpy_tags_tensor():
    prob=0.25
    tags = list(np.random.choice(2, len(misc),p=[1.0-prob,prob]))+[0 for i in range(len(tag_list)-len(misc))]
    for cat in [hair_type,hair_color,eyes_color]:
        choice = random.choice(cat)
        idx = tag_index[choice]
        tags[idx]=1
    return torch.from_numpy(np.array(tags)).float().view(1,-1)

def noise_sampler():
    fake_noise = Variable(torch.FloatTensor(batch_size, noise_input_dim)).cuda()
    fake_noise.data.normal_(0, 1)
    tags = torch.cat([get_random_numpy_tags_tensor() for i in range(batch_size)], dim=0)
    tags = Variable(tags).cuda()
    return fake_noise,tags


labels = Variable(torch.FloatTensor(batch_size,1)).cuda()

g = generator(tags=29).cuda()

checkpoint = torch.load('33_epoch_checkpoint.pth.tar')
g.load_state_dict(checkpoint['G'])
fake_noise,fake_tags = noise_sampler()
fake_data = torch.cat([fake_noise, fake_tags], dim= 1)
fake = g(fake_data)
vutils.save_image(fake.data.view(batch_size, 3, 128, 128),
        'fake_samples_.png'  )
