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
lambda_adv = 10
lambda_cl = 10
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

def save_checkpoint(state, filename='_epoch_checkpoint.pth.tar'):
    torch.save(state,str(state['epoch']) +filename)

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
        idx = tag_index[random.choice(cat)]
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
g.apply(weights_init)

d = discriminator(tags=29).cuda()
d.apply(weights_init)

d_optimizer = torch.optim.Adam(d.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
g_optimizer = torch.optim.Adam(g.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

anime_data = anime_face(at_p,image_dir,transform=transforms.Compose([ToTensor()]))
data_loader = torch.utils.data.DataLoader(anime_data,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4, drop_last =True)

criterion = nn.BCEWithLogitsLoss().cuda()
multi_tags_criterion = nn.MultiLabelSoftMarginLoss().cuda()
iterations=0
for epoch in range(max_epochs):
    adjust_learning_rate(g_optimizer, iterations)
    adjust_learning_rate(d_optimizer, iterations)
    for batch_idx, (image, tags) in enumerate(data_loader):
        image,tags = Variable(image).cuda(),Variable(tags).cuda()

        #real data training
        d.zero_grad()
        real_label_pred, real_tags_pred = d(image)
        labels.data.fill_(1.0)
        real_label_loss = criterion(real_label_pred, labels)
        real_tags_loss = multi_tags_criterion(real_tags_pred, tags)
        real_loss_sum = lambda_adv*real_label_loss+lambda_cl*real_tags_loss
        real_loss_sum.backward()

        #fake data training
        fake_noise,fake_tags = noise_sampler()
        fake_data = torch.cat([fake_noise, fake_tags],dim= 1)
        fake_data = g(fake_data).detach()
        fake_label_pred, fake_tags_pred = d(fake_data)
        labels.data.fill_(0.0)
        fake_label_loss = criterion(fake_label_pred, labels)
        fake_tags_loss = multi_tags_criterion(fake_tags_pred, fake_tags)
        fake_loss_sum = lambda_adv*fake_label_loss+lambda_cl*fake_tags_loss
        fake_loss_sum.backward()

        #dragan gradient penalty
        shape = [image.size(0)] + [1] * (image.dim() - 1)
        alpha = torch.rand(shape).cuda()
        pertubed_batch = image.data + 0.5 * image.data.std() *torch.rand(image.size()).cuda()
        differences = pertubed_batch-image.data
        interpolates = image.data + (alpha*differences)

        x_hat = Variable(interpolates, requires_grad=True)
        pred_hat, tags = d(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True)[0].view(x_hat.size(0),-1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()
       

        #update discriminator's weights
        loss_d = real_loss_sum + fake_loss_sum + gradient_penalty
        d_optimizer.step()     


        #train generator
        g.zero_grad()
        fake_noise,fake_tags = noise_sampler()
        fake_data = torch.cat([fake_noise, fake_tags],dim= 1)
        generated_fake_data = g(fake_data)
        label_pred, tags_pred = d(generated_fake_data)
        
        labels.data.fill_(1.0)
        
        generator_label_loss = criterion(label_pred, labels) 
        generator_tags_loss = multi_tags_criterion(fake_tags, tags_pred) 
        generator_loss = lambda_adv*generator_label_loss+lambda_cl*generator_tags_loss

        #update generator weights
        generator_loss.backward()
        g_optimizer.step()  

        if iterations % 100 == 0:

            vutils.save_image(image.data.view(batch_size, 3, 128, 128),
                    'samples/real_samples.png')
            fake_noise,fake_tags = noise_sampler()
            fake_data = torch.cat([fake_noise, fake_tags], dim= 1)
            fake = g(fake_data)
            vutils.save_image(fake.data.view(batch_size, 3, 128, 128),
                    'samples/fake_samples_iterations_%03d.png' % iterations)
        iterations+=1
    save_checkpoint({
        'epoch': epoch,
        'D': d.state_dict(),
        'G': g.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
    })
    print("saved")
            

