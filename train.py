
from discriminator import discriminator
from generator import generator
from data_loader import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
at_p='/home/elamuon/paper/tag_detection/illustration2vec/cleaned_image_atrri.p'
image_dir = '/home/elamuon/paper/tag_detection/illustration2vec/images/clean_resized_image/'
lambda_adv = 37
lambda_gp = 0.5
batch_size = 64
learning_rate=0.0002
beta_1 = 0.5
noise_input_dim = 128
max_epochs = 20
tag_list = [
'drill hair',
'twintails',
 'ponytail',
 'blush',
 'smile',
 'open mouth',
 'hat',
 'ribbon',
 'glasses',
 "1girl",
 "1boy",
'long hair', 'short hair', 
'blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair',
'blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes', 'grey eyes']
misc = ['drill hair','twintails','ponytail','blush','smile','open mouth','hat','ribbon','glasses']
sex = ["1girl","1boy"]
hair_type=['long hair', 'short hair'] 
hair_color= ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair']
eyes_color= ['blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes', 'grey eyes']
tag_index = {}
for i in range(len(tag_list)):
    tag_index[tag_list[i]] = i
def adjust_learning_rate(optimizer, iterations):
    """Sets the learning rate to the initial LR decayed by 10 every 50000 iterations"""
    lr = learning_rate * (0.1 ** (iterations // 50000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)

def noise_sampler():
    fake_noise = Variable(torch.FloatTensor(batch_size, noise_input_dim)).data.normal_(0, 1).cuda()
    prob=0.25
    tags = list(np.random.choice(2, len(misc),p=[1.0-prob,prob]))+[0 for i in range(len(tag_list)-len(misc))]
    for cat in [sex,hair_type,hair_color,eyes_color]:
        idx = tag_index[random.choice(cat)]
        tags[idx]=1
    tags = Variable(torch.from_numpy(np.array(tags)).float()).cuda()
    return fake_noise,tags




labels = Variable(torch.FloatTensor(batch_size,1)).cuda()

generator = generator(tags=37).cuda()
generator.apply(weights_init)

discriminator = discriminator(tags=37).cuda()
discriminator.apply(weights_init)

discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

anime_data = anime_face(at_p,image_dir,transform=transforms.Compose([ToTensor()]))
data_loader = torch.utils.data.DataLoader(anime_data,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)
criterion = torch.nn.BCEWithLogitsLoss().cuda()
iterations=0
for epoch in range(max_epochs):
    adjust_learning_rate(generator_optimizer, iterations)
    adjust_learning_rate(discriminator_optimizer, iterations)
    for batch_idx, minibatch_data in enumerate(data_loader):
        image,tags = minibatch_data['image'],minibatch_data['ouput']
        image,tags = Variable(image).cuda(),Variable(tags).cuda()
        #real data training
        discriminator.zero_grad()
        real_label_pred, real_tags_pred = discriminator(image)

        labels.data.fill_(1.0)

        real_label_loss = criterion(real_label_pred, labels)
        real_tags_loss = criterion(real_tags_pred, tags)
        real_loss_sum = lambda_adv*real_label_loss+real_tags_loss
        real_loss_sum.backward()

        #fake data training
        fake_noise,fake_tags = noise_sampler()
        fake_data = torch.cat((fake_noise, fake_tags), 1)
        fake_data = generator(fake_data).detach()
        fake_label_pred, fake_tags_pred = discriminator(fake_data)
        
        labels.data.fill_(0.0)
    
        fake_label_loss = criterion(fake_label_pred, labels)
        fake_tags_loss = criterion(fake_tags_pred, fake_tags)
        fake_loss_sum = lambda_adv*fake_label_loss+fake_tags_loss
        fake_loss_sum.backward()


       # gradient penalty
        alpha = torch.rand(batch_size, 1).expand(X.size())
        x_hat = Variable(alpha * X.data + (1 - alpha) * (X.data + 0.5 * X.data.std() * torch.rand(X.size())), requires_grad=True)
        pred_hat, tags = discriminator(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()

        #update discriminator's weights
        loss_d = real_loss_sum + fake_loss_sum + gradient_penalty
        discriminator_optimizer.step()     


        #train generator
        generator.zero_grad()
        fake_noise,fake_tags = noise_sampler()
        fake_data = torch.cat((fake_noise, fake_tags), 1)
        generated_fake_data = generator(fake_data)
        label_pred, tags_pred = discriminator(generated_fake_data)
        
        labels.data.fill_(1.0)
        
        generator_label_loss = criterion(label_pred, labels) 
        generator_tags_loss = criterion(fake_tags, tags_pred) 
        generator_loss = lambda_adv*generator_label_loss+generator_tags_loss
        generator_loss.backward()
        generator_optimizer.step()  

        iterations+=1

        # if print_every:
        #     print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
        #           % (epoch, max_epochs, batch_idx, len(train_loader),
        #              loss_d.data[0], loss_g.data[0]))

        #     if batch_idx % 100 == 0:
        #         vutils.save_image(data,
        #                 'samples_vanilla/real_samples.png')
        #         fake = generator(z)
        #         vutils.save_image(gen.data.view(batch_size, 1, 28, 28),
        #                 'samples_vanilla/fake_samples_epoch_%03d.png' % epoch)