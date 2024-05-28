'''
@author: Yang Hu
'''
''' ------------------ optimizers for all algorithms (models) ------------------ '''

import csv
import os

from accelerate.accelerator import Accelerator
from torch import optim, nn
import torch
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import networks
from models.networks import clip_loss
import numpy as np
from support import env
from support.env import ENV
from support.tools import Time


def optimizer_sgd_basic(net, lr=1e-2):
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    return optimizer, scheduler


def optimizer_adam_basic(net, lr=1e-3, wd=1e-2):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    return optimizer

def optimizer_adam_w(net, lr=1e-3, wd=1e-2):
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


''' ------------- training function for CLIP ------------ '''

def train_clip(model, dataloader, epochs, optimizer, store_path,
               grad_clip=False, milestons=[0.2, 0.4, 0.6, 0.8, 1.0]):
    '''
    '''
    
    model = env._todevice(model)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        dataloader.dataset.reset_shuffling()
        model_path = store_path
        
        run_time = Time()
        for batch in dataloader:
        # for batch in tqdm(dataloader, desc=f"SP Epoch {epoch+1}/{epochs}"):
            img_small = env._todevice(batch['img_small'])
            img_large = env._todevice(batch['img_large'])
            gene_ids = env._todevice(batch['gene_ids'])
            gene_exp = env._todevice(batch['gene_exp'])
            mask = env._todevice(batch['mask'])
            
            optimizer.zero_grad()
            
            image_features, gene_features = model(img_small, img_large, gene_ids, gene_exp, mask)
            loss = clip_loss(image_features, gene_features, model.temperature)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # grad cut
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"SP Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, running time: {str(run_time.elapsed())[:-5]}")
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        check_points = [int(epochs * ms) for ms in milestons]
        if epoch + 1 in check_points or epoch + 1 >= epochs:
            model_path = model_path.replace('.pth', f'[{epoch + 1}].pth')
            _ = networks.store_net(model, model_path, optimizer)
            print(f'record checkpoint at: {model_path}')
            
def train_clip_multi_gpu_torch(model, dataloader, epochs, optimizer, store_path,
                               grad_clip=False, milestons=[0.2, 0.4, 0.6, 0.8, 1.0]):
    # Move model to the available GPU(s)
    model = nn.DataParallel(model)  # Use DataParallel to utilize multiple GPUs
    model = env._todevice(model)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        dataloader.dataset.reset_shuffling()
        model_path = store_path
        
        run_time = Time()
        for batch in dataloader:
        # for batch in tqdm(dataloader, desc=f"MP-T Epoch {epoch+1}/{epochs}"):
            img_small = env._todevice(batch['img_small'])
            img_large = env._todevice(batch['img_large'])
            gene_ids = env._todevice(batch['gene_ids'])
            gene_exp = env._todevice(batch['gene_exp'])
            mask = env._todevice(batch['mask'])
            
            optimizer.zero_grad()
            image_features, gene_features = model(img_small, img_large, gene_ids, gene_exp, mask)
            loss = clip_loss(image_features, gene_features, model.module.temperature)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # grad cut
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"MP-T Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, running time: {str(run_time.elapsed())[:-5]}")
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        check_points = [int(epochs * ms) for ms in milestons]
        if epoch + 1 in check_points or epoch + 1 >= epochs:
            model_path = model_path.replace('.pth', f'[{epoch + 1}].pth')
            _ = networks.store_net(model, model_path, optimizer)
            print(f'record checkpoint at: {model_path}')
            
        
def train_clip_multi_gpu(model, dataloader, epochs, optimizer, store_path,
                         milestons=[0.2, 0.4, 0.6, 0.8, 1.0]):
    '''
    TODO: not work on multi-GPU, need to check
    '''
    
    accelerator = Accelerator()
    model.train()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch in range(epochs):
        total_loss = 0.0
        dataloader.dataset.reset_shuffling()
        model_path = store_path
        
        run_time = Time()
        for batch in dataloader:
        # for batch in tqdm(dataloader, desc=f"MP Epoch {epoch+1}/{epochs}"):
            img_small = batch['img_small']
            img_large = batch['img_large']
            gene_ids = batch['gene_ids']
            gene_exp = batch['gene_exp']
            mask = batch['mask']
            
            optimizer.zero_grad()
            image_features, gene_features = model(img_small, img_large, gene_ids, gene_exp, mask)
            loss = clip_loss(image_features, gene_features, model.temperature)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"MP-T Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, running time: {str(run_time.elapsed())[:-5]}")
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        check_points = [int(epochs * ms) for ms in milestons]
        if epoch + 1 in check_points or epoch + 1 >= epochs:
            model_path = model_path.replace('.pth', f'[{epoch + 1}].pth')
            _ = networks.store_net(model, model_path, optimizer)
            print(f'record checkpoint at: {model_path}')



if __name__ == '__main__':
    pass