## 说明：1）在目标域单纯训练分类器，测试分类结果

import argparse
import os
import numpy as np
from torch.autograd import Variable

# from mnistm import MNISTM
from myDataloader import get_loader
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter###
# CUDA_VISIBLE_DEVICES=0
os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_classes', type=int, default=7, help='number of classes in the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')

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

# Initialize generator and discriminator
classifier = Classifier()
task_loss = torch.nn.CrossEntropyLoss()

if cuda:
    classifier.cuda()
    task_loss.cuda()
# Initialize weights
classifier.apply(weights_init_normal)
########
writer = SummaryWriter('runs')
# # model = LeNet()
# dummy_input = torch.rand(64, 3, 32, 32) #假设输入13张1*28*28的图片
# with SummaryWriter(comment='classifier') as w:
#     w.add_graph(classifier, (dummy_input, ))
#E:/Mydocuments/coders/DATASETS/RAF_DB/basic/images_aligned/train
#/data/dm/data/RAF_DB/train
dataloader_B_train = get_loader('/data/dm/data/RAF_DB/train',
                          crop_size=100, image_size=opt.img_size, batch_size=opt.batch_size,
                          num_workers=1)
dataloader_B_test = get_loader('/data/dm/data/RAF_DB/test',
                                crop_size=100, image_size=opt.img_size, batch_size=opt.batch_size,
                                num_workers=1)
# Optimizers
optimizer_C = torch.optim.Adam( classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
# ----------
#  Training
# ----------

# Keeps 100 accuracy measurementsc
train_performance = []
test_performance = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_B, labels_B), (imgs_B_test, labels_B_test)) in enumerate(zip(dataloader_B_train, dataloader_B_test)):

        batch_size = imgs_B.size(0)
        # Configure input
        imgs_B      = Variable(imgs_B.type(FloatTensor))
        labels_B    = Variable(labels_B.type(LongTensor))
        imgs_B_test      = Variable(imgs_B_test.type(FloatTensor))
        labels_B_test    = Variable(labels_B_test.type(LongTensor))
        optimizer_C.zero_grad()
        label_predB = classifier(imgs_B)
        c_loss = task_loss(label_predB, labels_B)
        # g_loss =    lambda_adv * adversarial_loss(discriminator(fake_B), valid) + \
        #             lambda_task * task_loss_ + lambda_rec * g_loss_rec
        c_loss.backward()
        optimizer_C.step()
        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on Domain B
        train_acc = np.mean(np.argmax(label_predB.data.cpu().numpy(), axis=1) == labels_B.data.cpu().numpy())
        train_performance.append(train_acc)
        if len(train_performance) > 100:
            train_performance.pop(0)

        label_predB_test = classifier(imgs_B_test)
        test_acc = np.mean(np.argmax(label_predB_test.data.cpu().numpy(), axis=1) == labels_B_test.data.cpu().numpy())
        test_performance.append(test_acc)
        if len(test_performance) > 100:
            test_performance.pop(0)

        print ("[Epoch %d/%d] [Batch %d/%d] ], train_acc: %3d%% (%3d%%), test_acc: %3d%% (%3d%%)]" %
               (epoch, opt.n_epochs,
                i, len(dataloader_B_train),
                100*train_acc, 100*np.mean(train_performance),
                100*test_acc, 100*np.mean(test_performance)))

    writer.add_scalar('c_loss', c_loss, epoch)
    writer.add_scalar('train_acc', 100*train_acc, epoch)
    writer.add_scalar('train_acc_mean', 100*np.mean(train_performance), epoch)
    writer.add_scalar('test_acc', 100*test_acc, epoch)
    writer.add_scalar('test_acc_mean', 100*np.mean(test_performance), epoch)################
writer.close()###################