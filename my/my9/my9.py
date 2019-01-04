## 说明：1）pixelda数据库MNIST->MNIST_M手写体换成人脸表情BU3D->RAF_DB
#       2）pixelda中的D,G换成StarGAN中的D,G,包括D,G的输入（带标签编码信息的输入数据）（未完成）
#       3）pixelda中的G的输入，换成跟stargan一样的输入，即人脸表情图片和两个域的类别标签掩码
#         （已完成，只要将lantend-dim（z）变成相应维度即可）
#       4) 学习率lr=0.0001（原先0.0002）
#       5)epoch=500(原先200->2000->500)
#       6)标签z=[c_org,c_trg,mask],图像：A,Fake_B,rec_BA,B
#       6)标签z=[z,c_org,c_trg,mask],图像：A,Fake_B,B,学习率lr=0.00001（原先0.0001）
#       6)标签z=[c_org,mask],图像：A,Fake_B,B,学习率lr=0.00001,有重构

import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# from mnistm import MNISTM
import sys
sys.path.append("/home/glj/codes/dm/my")
from myDataloader import get_loader
import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter###
# CUDA_VISIBLE_DEVICES=0
os.makedirs('images9', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_residual_blocks', type=int, default=6, help='number of residual blocks in generator')
parser.add_argument('--latent_dim', type=int, default=9, help='dimensionality of the noise input')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_classes', type=int, default=7, help='number of classes in the dataset')
parser.add_argument('--sample_interval', type=int, default=300, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)
# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2**4)
patch = (1, patch, patch)
cuda = True if torch.cuda.is_available() else False

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels*opt.img_size**2)
        self.l1 = nn.Sequential(nn.Conv2d(opt.channels*2, 64, 3, 1, 1), nn.ReLU(inplace=True))
        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)
        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())
    def forward(self, img, z):
        # test =self.fc(z).view(*img.shape)
        # print (test.dtype)
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)
        return img_

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True) ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True) ]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512)
        )

        input_size = opt.img_size // 2**4
        self.output_layer = nn.Sequential(
            nn.Linear(512*input_size**2, opt.n_classes),
            nn.Softmax()
        )

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label

# Loss function
adversarial_loss = torch.nn.MSELoss()
task_loss = torch.nn.CrossEntropyLoss()

# Loss weights
lambda_adv =  1
lambda_task = 0.1
lambda_rec = 0.1

# Initialize generator and discriminator
generator= Generator()
discriminator = Discriminator()
classifier = Classifier()

if cuda:
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    adversarial_loss.cuda()
    task_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)
########
writer = SummaryWriter('runs9')
# # model = LeNet()
# dummy_input = torch.rand(64, 3, 32, 32) #假设输入13张1*28*28的图片
# with SummaryWriter(comment='generator') as w:
#     w.add_graph(generator, (dummy_input, ))
dataloader_A = get_loader('/data/dm/data/BU_3DFE/train',crop_size=900,
                          image_size=opt.img_size, batch_size=opt.batch_size,num_workers=1)
dataloader_B = get_loader('/data/dm/data/RAF_DB/train',crop_size=100,
                          image_size=opt.img_size, batch_size=opt.batch_size,num_workers=1)
# Optimizers
optimizer_G = torch.optim.Adam( itertools.chain(generator.parameters(), classifier.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
# g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurementsc
source_performance = []
task_performance = []
target_performance = []
s_acc = []
f_B_acc = []
t_acc = []

for epoch in range(opt.n_epochs):
    # for dataset in ['A', 'B']:
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):

        batch_size = imgs_A.size(0)
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)
        # Configure input
        imgs_A      = Variable(imgs_A.type(FloatTensor))
        imgs_A      = Variable(imgs_A.type(FloatTensor).expand(batch_size, 3, opt.img_size, opt.img_size))
        labels_A    = Variable(labels_A.type(LongTensor))
        imgs_B      = Variable(imgs_B.type(FloatTensor))
        # labels_B    = Variable(labels_B.type(LongTensor))

        c_org = label2onehot(labels_A, opt.n_classes)
        # c_trg = label2onehot(labels_B, opt.n_classes)
        # zero = torch.zeros(imgs_A.size(0), 7)
        mask_A = label2onehot(torch.zeros(imgs_A.size(0)), 2)
        mask_B = label2onehot(torch.ones(imgs_B.size(0)), 2)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Sample noise
        # z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.n_classes))))
        z = Variable(torch.cat([c_org,mask_A],dim=1))
        z = z.cuda()
        # Generate a batch of images
        fake_B = generator(imgs_A, z)
        # Perform task on translated source image
        label_predB = classifier(fake_B)
        ##
        # z_r = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.n_classes))))
        z_rec = Variable(torch.cat([c_org,mask_B],dim=1))
        z_rec = z_rec.cuda()
        fake_BA =generator(fake_B,z_rec)
        label_predBA = classifier(fake_BA)
        g_loss_rec = torch.mean(torch.abs(imgs_A - fake_BA))
        ##

        # Calculate the task loss
        task_loss_ =    (task_loss(label_predB, labels_A) + \
                         task_loss(label_predBA, labels_A) + \
                         task_loss(classifier(imgs_A), labels_A)) / 3

        # Loss measures generator's ability to fool the discriminator
        # discriminator_fake_B=discriminator(fake_B)
        g_loss =    lambda_adv * adversarial_loss(discriminator(fake_B), valid) + \
                    lambda_task * task_loss_ + lambda_rec * g_loss_rec

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_loss_real = adversarial_loss(discriminator(imgs_B), valid)
        d_loss_fake = adversarial_loss(discriminator(fake_B.detach()), fake)
        # d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss = (d_loss_real + d_loss_fake)/2

        d_loss.backward()
        optimizer_D.step()
        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        S_acc = np.mean(np.argmax(classifier(imgs_A).data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        source_performance.append(S_acc)
        if len(source_performance) > 100:
            source_performance.pop(0)

        #Evaluate performance on translated Fake B
        acc = np.mean(np.argmax(label_predB.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.data.cpu().numpy())
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF S_acc: %3d%% (%3d%%), predB_acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]" %
               (epoch, opt.n_epochs,
                i, len(dataloader_A),
                d_loss.data[0], g_loss.data[0],
                100*S_acc, 100*np.mean(source_performance),
                100*acc, 100*np.mean(task_performance),
                100*target_acc, 100*np.mean(target_performance)))

        batches_done = len(dataloader_A) * epoch + i
        if batches_done % opt.sample_interval == 0:
            # sample = torch.cat((imgs_A.data[:5], fake_B.data[:5], fake_BA.data[:5], imgs_B.data[:5]), -2)
            sample = torch.cat((imgs_A.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            save_image(sample, 'images9/%d.png' % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)
            # Evaluate performance on translated Domain A

    s_acc.append(S_acc)
    f_B_acc.append(acc)
    t_acc.append(target_acc)

    print ("[Epoch %d/%d] [CLF S_acc: %3d%%, predB_acc: %3d%%, target_acc: %3d%% " %
           (epoch, opt.n_epochs,100*np.mean(s_acc), 100*np.mean(f_B_acc),100*np.mean(t_acc)))
    writer.add_scalar('g_loss', g_loss, epoch)
    writer.add_scalar('g_loss_rec', g_loss_rec, epoch)
    writer.add_scalar('real_loss', d_loss_real, epoch)
    writer.add_scalar('fake_loss', d_loss_fake, epoch)
    writer.add_scalar('d_loss', d_loss, epoch)
    writer.add_scalar('task_loss', task_loss_, epoch)
    writer.add_scalar('source_performance_acc', 100*S_acc, epoch)
    writer.add_scalar('source_performance_acc_mean', 100*np.mean(source_performance), epoch)
    writer.add_scalar('task_performance_acc', 100*acc, epoch)
    writer.add_scalar('task_performance_acc_mean', 100*np.mean(task_performance), epoch)
    writer.add_scalar('target_acc', 100*target_acc, epoch)
    writer.add_scalar('target_acc_mean', 100*np.mean(target_performance), epoch)
    writer.add_scalar('s_acc_mean', 100*np.mean(s_acc), epoch)
    writer.add_scalar('f_B_acc', 100*np.mean(f_B_acc), epoch)
    writer.add_scalar('t_acc_mean', 100*np.mean(t_acc), epoch)################
writer.close()###################

test_performance = []
# classifier = torch.load('classifer.pkl')
dataloader_B_test = get_loader('/data/dm/data/RAF_DB/test',
                               crop_size=100, image_size=opt.img_size, batch_size=opt.batch_size,
                               num_workers=1)
for images, labels in dataloader_B_test:
    images = Variable(images.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))
    pred = classifier(images)
    pred_acc = np.mean(np.argmax(pred.data.cpu().numpy(), axis=1) == labels.cpu().numpy())
    test_performance.append(pred_acc)
print('Accuracy of the model on the test images: %d %%' % (100*np.mean(test_performance)))