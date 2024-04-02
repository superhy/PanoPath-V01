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
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
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
        mean=[0.5, 0.5, 0.5],
        std=[0.1, 0.1, 0.1]
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


'''
----------------- pre-training functions -----------------
'''
def dino_epoch(learner, train_loader, optimizer, epoch_info: tuple=(-2, -2)):
    """
    self-supervised pre-training epoch with Dino
    
    Args:
        learner:
        train_loader:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    learner.train()
    epoch_loss_sum, batch_count, time = 0.0, 0, Time()
    
    for X in train_loader:
        X = X.cuda()
        # feed forward
        batch_loss = learner(X)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        learner.update_moving_average()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        batch_count += 1
    
    epoch_log = 'epoch [%d/%d], batch_loss-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                    epoch_loss_sum / batch_count,
                                                                    str(time.elapsed())[:-5])
    return epoch_log

def mae_epoch(learner, train_loader, optimizer, epoch_info: tuple=(-2, -2)):
    """
    self-supervised pre-training epoch with MAE
    
    Args:
        learner:
        train_loader:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    learner.train()
    epoch_loss_sum, batch_count, time = 0.0, 0, Time()
    
    for X in train_loader:
        X = X.cuda()
        # feed forward
        batch_loss = learner(X)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        batch_count += 1
    
    epoch_log = 'epoch [%d/%d], batch_loss-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                    epoch_loss_sum / batch_count,
                                                                    str(time.elapsed())[:-5])
    return epoch_log

'''
----------------- classification training/testing functions -----------------
'''
def patch_cls_train_epoch(net, train_loader, criterion, optimizer, epoch_info: tuple=(-2, -2)):
    """
    classification training epoch with any network
    
    Args:
        learner:
        train_loader:
        criterion: loss function
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    net.train()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    for X, y in train_loader:
        X = X.cuda()
        y = y.cuda()
        # feed forward
        y_pred = net(X)
        batch_loss = criterion(y_pred, y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
        
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, train acc (on tiles)-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                                 epoch_loss_sum / batch_count,
                                                                                                 epoch_acc_sum / len(train_loader.dataset),
                                                                                                 str(time.elapsed())[:-5])
    return train_log

def patch_cls_test_epoch(net, test_loader, criterion):
    """
    classification testing epoch with any network
    PS: remove acc reporting in test_log from this project
    
    Args:
        learner:
        train_loader:
        criterion: loss function
    """
    net.eval()
    epoch_loss_sum, batch_count, time = 0.0, 0, Time()
    
    y_pred_scores, y_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.cuda()
            y = y.cuda()
            # feed forward
            y_pred = net(X)
            batch_loss = criterion(y_pred, y)
            # loss count
            epoch_loss_sum += batch_loss.cpu().item()
            y_pred = softmax(y_pred, dim=-1)
#             epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            batch_count += 1
            
            y_pred_scores.extend(y_pred.detach().cpu().numpy()[:, -1].tolist())
            y_labels.extend(y.cpu().numpy().tolist())
            
    test_log = 'test loss-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count, str(time.elapsed())[:-5])
    
    return test_log, np.array(y_pred_scores), np.array(y_labels)


def train_agt_epoch(net, train_loader, loss, optimizer, epoch_info: tuple=(-2, -2)):
    """
    trainer for slide-level(WSI) feature aggregation and/or classification
    
    Args:
        net: diff with other training function, net need to input <mat_X, bag_dim>
        data_loader:
        loss:
        optimizer:
        epoch: the idx of running epoch (default: None (unknown))
    """
    net.train()
    epoch_loss_sum, epoch_acc_sum, batch_count, time = 0.0, 0.0, 0, Time()
    
    for mat_X, bag_dim, y in train_loader:
        mat_X = mat_X.cuda()
        bag_dim = bag_dim.cuda()
        y = y.cuda()
        # feed forward
        y_pred, _, _ = net(mat_X, bag_dim)
        batch_loss = loss(y_pred, y)
        # BP
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # loss count
        epoch_loss_sum += batch_loss.cpu().item()
        epoch_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        batch_count += 1
    
#     train_log = 'batch_loss-> %.6f, train acc-> %.4f, time: %s sec' % (epoch_loss_sum / batch_count, epoch_acc_sum / len(train_loader.dataset), str(time.elapsed()))
    train_log = 'epoch [%d/%d], batch_loss-> %.4f, train acc-> %.4f, time: %s sec' % (epoch_info[0] + 1, epoch_info[1],
                                                                                      epoch_loss_sum / batch_count,
                                                                                      epoch_acc_sum / len(train_loader.dataset),
                                                                                      str(time.elapsed())[:-5])
    return train_log

''' ------------- evaluation methods ------------- '''
def regular_evaluation(y_scores, y_label):
    '''
    balanced_accuracy and auc
    '''
    y_pred = np.array([1 if score > 0.5 else 0 for score in y_scores.tolist()])
    acc = metrics.balanced_accuracy_score(y_label, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(y_label, y_scores)
    auc = metrics.auc(fpr, tpr)
        
    return acc, fpr, tpr, auc

def recall_evaluation(y_scores, y_label, record_auc=True):
    '''
    recall on class of roi region and auc
    '''
    y_pred = np.array([1 if score > 0.5 else 0 for score in y_scores.tolist()])
    recall = metrics.recall_score(y_label, y_pred)
    if record_auc:
        fpr, tpr, threshold = metrics.roc_curve(y_label, y_scores)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = None
    
    return recall, fpr, tpr, auc
        
def f1_evaluation(y_scores, y_label, record_auc=True):
    '''
    f1 on class of roi region and auc
    '''
    y_pred = np.array([1 if score > 0.5 else 0 for score in y_scores.tolist()])
    f1 = metrics.f1_score(y_label, y_pred, pos_label=1) # only consider the roi part
    if record_auc:
        fpr, tpr, threshold = metrics.roc_curve(y_label, y_scores)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = None
    
    return f1, fpr, tpr, auc

def p_r_f1_evaluation(y_scores, y_label, record_auc=True):
    '''
    precision, recall, f1 on class of roi region and auc
    '''
    y_pred = np.array([1 if score > 0.5 else 0 for score in y_scores.tolist()])
    
    print(y_pred)
    print(y_label)
    nb_p_0, nb_p_1, nb_p_x = 0, 0, 0
    nb_a_0, nb_a_1, nb_a_x = 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            nb_p_0 += 1
        elif y_pred[i] == 1:
            nb_p_1 += 1
        else:
            nb_p_x +=1
    
        if y_label[i] == 0:
            nb_a_0 += 1
        elif y_label[i] == 1:
            nb_a_1 += 1
        else:
            nb_a_x +=1
    print(nb_p_0, nb_p_1, nb_p_x)
    print(nb_a_0, nb_a_1, nb_a_x) 
    
    precision = metrics.precision_score(y_label, y_pred)
    recall = metrics.recall_score(y_label, y_pred)
    f1 = metrics.f1_score(y_label, y_pred)
    precisions, recalls, _ = metrics.precision_recall_curve(y_label, y_scores, pos_label=1)
    prc_pkg = (precisions, recalls)
    if record_auc:
        fpr, tpr, _ = metrics.roc_curve(y_label, y_scores)
        auc = metrics.auc(fpr, tpr)
        roc_pkg = (fpr, tpr, auc)
    else:
        roc_pkg = None
        
    return precision, recall, f1, prc_pkg, roc_pkg

def store_evaluation_roc(csv_path, roc_set):
    '''
    store the evaluation results as ROC as csv file
    '''
    acc, fpr, tpr, auc = roc_set
    with open(csv_path, 'w', newline='') as record_file:
        csv_writer = csv.writer(record_file)
        csv_writer.writerow(['acc', 'auc', 'fpr', 'tpr'])
        for i in range(len(fpr)):
            csv_writer.writerow([acc, auc, fpr[i], tpr[i]])
    print('write roc record: {}'.format(csv_path))

if __name__ == '__main__':
    pass