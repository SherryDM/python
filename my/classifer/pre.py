## 说明：1）在源域(BU3D_FE)上训练分类器，保存模型参数，在目标域(RAF_DB)测试集上测试分类结果

import argparse
import os
import numpy as np
from torch.autograd import Variable

# from mnistm import MNISTM
import sys
sys.path.append("/home/glj/codes/dm/my")
from myDataloader import get_loader
import torch.nn as nn
import torch
# import torch.nn.Module
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
dataloader_A = get_loader('/data/dm/data/BU_3DFE/train',
                                crop_size=100, image_size=opt.img_size, batch_size=opt.batch_size,
                                num_workers=1)
dataloader_B = get_loader('/data/dm/data/RAF_DB/train',
                               crop_size=100, image_size=opt.img_size, batch_size=opt.batch_size,
                               num_workers=1)
# Optimizers
optimizer_C = torch.optim.Adam( classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
# -------------------------------------------------
#              Training
# -------------------------------------------------

# Keeps 100 accuracy measurementsc
train_performance = []
valid_performance = []

for epoch in range(opt.n_epochs):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):
        batch_size = imgs_B.size(0)
        # Configure input
        imgs_A      = Variable(imgs_A.type(FloatTensor))
        labels_A    = Variable(labels_A.type(LongTensor))
        imgs_B      = Variable(imgs_B.type(FloatTensor))
        labels_B    = Variable(labels_B.type(LongTensor))
        optimizer_C.zero_grad()
        label_predA = classifier(imgs_A)
        c_loss = task_loss(label_predA, labels_A)
        # g_loss =    lambda_adv * adversarial_loss(discriminator(fake_B), valid) + \
        #             lambda_task * task_loss_ + lambda_rec * g_loss_rec
        c_loss.backward()
        optimizer_C.step()
        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------
        # Evaluate performance on Domain B
        train_acc = np.mean(np.argmax(label_predA.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        train_performance.append(train_acc)
        if len(train_performance) > 100:
            train_performance.pop(0)

        label_predB_valid = classifier(imgs_B)
        valid_acc = np.mean(np.argmax(label_predB_valid.data.cpu().numpy(), axis=1) == labels_B.data.cpu().numpy())
        valid_performance.append(valid_acc)
        if len(valid_performance) > 100:
            valid_performance.pop(0)

        print ("[Epoch %d/%d] [Batch %d/%d] ], train_acc: %3d%% (%3d%%), test_acc: %3d%% (%3d%%)]" %
               (epoch, opt.n_epochs,
                i, len(dataloader_A),
                100*train_acc, 100*np.mean(train_performance),
                100*valid_acc, 100*np.mean(valid_performance)))

    writer.add_scalar('c_loss', c_loss, epoch)
    writer.add_scalar('train_acc', 100*train_acc, epoch)
    writer.add_scalar('train_acc_mean', 100*np.mean(train_performance), epoch)
    writer.add_scalar('valid_acc', 100*valid_acc, epoch)
    writer.add_scalar('valid_acc_mean', 100*np.mean(valid_performance), epoch)################
writer.close()###################
# Save the Model
torch.save(classifier.state_dict(), 'classifer.pkl')

#------------------------------------------------
#                test
# -----------------------------------------------
test_performance = []
classifier = Classifier()
classifier.cuda()
classifier.load_state_dict(torch.load('classifer.pkl'))
dataloader_B_test = get_loader('/data/dm/data/RAF_DB/test',
                               crop_size=100, image_size=opt.img_size, batch_size=opt.batch_size,
                               num_workers=1)
for i,(images, labels) in enumerate(dataloader_B_test):
    images = Variable(images.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))
    pred = classifier(images)
    pred_acc = np.mean(np.argmax(pred.data.cpu().numpy(), axis=1) == labels.cpu().numpy())
    test_performance.append(pred_acc)
print('Accuracy of the model on the test images: %d %%' % (100*np.mean(test_performance)))