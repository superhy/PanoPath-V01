'''
@author: Yang Hu
'''
import csv
import os

from sklearn import metrics
from torch import nn, optim
import torch
from torch.nn.functional import softmax
from torch.nn.modules.loss import CrossEntropyLoss, L1Loss, NLLLoss, \
    TripletMarginLoss
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import numpy as np
from support.env import ENV
from support.tools import Time


#######################################################################
#------------- a list of self-designed losses (criterion) ------------#
#######################################################################
class CombinationLoss(nn.Module):
    
    def __init__(self, nb_losses, loss_lambda: list=[0.5, 0.5]):
        super(CombinationLoss, self).__init__()
        self.weights = []
        self.left_lambda = 1.0
        if loss_lambda != None:
            i = 0
            for _lambda in loss_lambda:
                self.weights.append(_lambda)
                self.left_lambda -= _lambda
                i += 1
            if i < nb_losses:
                self.weights.extend(list(max(self.left_lambda, 0) / (nb_losses - i) for j in range(nb_losses - i)))
#             para = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#             para = torch.clamp(para, min=0.0, max=1.0)
#             self.weights.append(para)
#             self.weights.append(1.0 - para)
        else:
            for i in range(nb_losses):
                self.weights.append(1.0)
                
    def forward(self, _losses):
        '''
        Args:
            _losses: multiple computed losses
        '''
        comb_loss = self.weights[0] * _losses[0]
        for i in range(len(_losses) - 1):
            comb_loss = comb_loss + self.weights[i + 1] * _losses[i + 1]
            
        return comb_loss


'''
------------- call various loss functions ------------
'''

def l1_loss():
    return L1Loss().cuda()

def nll_loss():
    return NLLLoss().cuda()

def cel_loss():
    return CrossEntropyLoss().cuda()

def weighted_cel_loss(weight=0.5):
    w = torch.Tensor([1 - weight, weight])
    loss = CrossEntropyLoss(w).cuda()
    return loss

def triplet_margin_loss():
    return TripletMarginLoss(margin=1.0, p=2).cuda()

def combination_loss(n_losses, loss_lambda=[0.5, 0.5]):
    return CombinationLoss(n_losses, loss_lambda).cuda()

''' ------------------ optimizers for all algorithms (models) ------------------ '''


def optimizer_sgd_basic(net, lr=1e-2):
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    return optimizer, scheduler


def optimizer_adam_basic(net, lr=1e-4, wd=1e-4):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def optimizer_rmsprop_basic(net, lr=1e-5):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    return optimizer


def optimizer_adam_pretrained(net, lr=1e-4, wd=1e-4):
    output_params = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, net.parameters())
    
    optimizer = optim.Adam([{'params': feature_params},
                            {'params': net.fc.parameters(), 'lr': lr * 1}],
                            lr=lr, weight_decay=wd)
    return optimizer

''' ------------------ dataloader ------------------ '''

def get_data_loader(dataset, batch_size, num_workers=4, sf=False, p_mem=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, 
                             num_workers=num_workers, 
                             shuffle=sf, 
                             pin_memory=p_mem)
    return data_loader


''' 
------------------ 
data transform with for loading batch data,
with / without data augmentation
------------------ 
'''

def get_transform():
    '''
    data transform with only image normalization
    '''
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_augs = transforms.Compose([
        transforms.Resize(size=(ENV.TRANSFORMS_RESIZE, ENV.TRANSFORMS_RESIZE)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs

def get_redu_size_transform():
    '''
    data transform with image normalization
    and size reducing
    '''
    redu_size = int(ENV.TRANSFORMS_RESIZE / 2)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_augs = transforms.Compose([
        transforms.Resize(size=(redu_size, redu_size)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs
    
def get_data_arg_transform():
    '''
    data transform with slight data augumentation
    '''
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
#     _resize = (256, 256) if ENV.TRANSFORMS_RESIZE < 300 else (ENV.TRANSFORMS_RESIZE - 20, TRANSFORMS_RESIZE - 20)
    transform_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=ENV.TRANSFORMS_RESIZE, scale=(0.8, 1.0)),
#         transforms.CenterCrop(size=_resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=15),
        transforms.Resize(size=(ENV.TRANSFORMS_RESIZE, ENV.TRANSFORMS_RESIZE)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs

def get_data_arg_redu_size_transform():
    '''
    data transform with slight data augumentation
    and size reducing
    '''
    redu_size = int(ENV.TRANSFORMS_RESIZE / 2)
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
    )
#     _resize = (256, 256) if ENV.TRANSFORMS_RESIZE < 300 else (ENV.TRANSFORMS_RESIZE - 20, TRANSFORMS_RESIZE - 20)
    transform_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=ENV.TRANSFORMS_RESIZE, scale=(0.8, 1.0)),
#         transforms.CenterCrop(size=_resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=15),
        transforms.Resize(size=(redu_size, redu_size)),
        transforms.ToTensor(),
        normalize
        ])
    return transform_augs



if __name__ == '__main__':
    pass