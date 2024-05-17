'''
Created on 14 May 2024

@author: yang hu
'''


import torch
from transformers import ViTModel, ViTConfig
from vit_pytorch.extractor import Extractor
from vit_pytorch.recorder import Recorder
from vit_pytorch.vit import ViT

import torch.nn as nn

from torch.nn.modules.loss import CrossEntropyLoss, L1Loss, NLLLoss, \
    TripletMarginLoss


"""
This file include networks for modelling images and cross-modalities
"""

class ContextSmallViT(nn.Module):
    """
    Vision Transformer using vit-pytorch for tissue image analysis with specified output dimension.
    Small ViT based on vit-pytorch is without pre-training
    """
    
    def __init__(self, image_size=224, patch_size=16, heads=4, depth=3, mlp_dim=256, hidden_dim=128):
        super(ContextSmallViT, self).__init__()
        
        self.network_name = f'CtxSmallViT{patch_size}-{image_size}_h{heads}_d{depth}'

        # Define the shared ViT model without a classification head
        self.vit_shared = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=0,  # This should be set to 0 if we only need features
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.2,
            emb_dropout=0.2
        )
        self.with_wrapper = False
        
        self.vit_shared.mlp_head = nn.Identity()

        # Using a layer to combine the features of small and large images
        self.encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Using hidden_dim now for the combination
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def deploy_recorder(self):
        self.backbone = Recorder(self.backbone)
        self.with_wrapper = True
    
    def deploy_extractor (self):
        self.backbone = Extractor(self.backbone)
        self.with_wrapper = True
    
    def discard_wrapper(self):
        self.backbone.eject()
        self.with_wrapper = False

    def forward(self, img_small, img_large):
        outputs_small = self.vit_shared(img_small)  # Get the transformer features
        outputs_large = self.vit_shared(img_large)
        
        # Map to the specified dimension
        # outputs_small = self.to_features(outputs_small)
        # outputs_large = self.to_features(outputs_large)

        # Combine the features
        combined_features = torch.cat([outputs_small, outputs_large], dim=-1)
        
        # Pass through the encoder
        output = self.encoder(combined_features)
        return output

class ContextShareViT(nn.Module):
    """
    """
    
    def __init__(self, hidden_dim=128, model_name='WinKawaks/vit-tiny-patch16-224'):
        super(ContextShareViT, self).__init__()
        
        self.network_name = f'CtxShareViT_{model_name}'
        
        # load the pre-trained ViT from HuggingFace
        config_img = ViTConfig.from_pretrained(model_name)
        config_img.num_labels = hidden_dim
        
        self.vit_shared = ViTModel(config_img)
        
        # using a layer to combine the features of small and large images
        self.encoder = nn.Sequential(
            nn.Linear(2 * config_img.hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, img_small, img_large):
        outputs_small = self.vit_shared(pixel_values=img_small).last_hidden_state[:, 0]
        outputs_large = self.vit_shared(pixel_values=img_large).last_hidden_state[:, 0]
        
        # combine the feature
        combined_features = torch.cat([outputs_small, outputs_large], dim=-1)
        
        # get through the encoder
        output = self.encoder(combined_features)
        return output

class ContextDualViT(nn.Module):
    """
    """
    
    def __init__(self, hidden_dim=128, model_name='WinKawaks/vit-tiny-patch16-224'):
        super(ContextDualViT, self).__init__()
        
        self.network_name = 'ContextDualViT'
        
        # load the pre-trained ViT from HuggingFace
        config_img = ViTConfig.from_pretrained(model_name)
        config_img.num_labels = hidden_dim
        
        self.vit_small = ViTModel(config_img)
        self.vit_large = ViTModel(config_img)
        
        # using a layer to combine the features of small and large images
        self.encoder = nn.Sequential(
            nn.Linear(2 * config_img.hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, img_small, img_large):
        outputs_small = self.vit_small(pixel_values=img_small).last_hidden_state[:, 0]
        outputs_large = self.vit_large(pixel_values=img_large).last_hidden_state[:, 0]
        
        # combine the feature
        combined_features = torch.cat([outputs_small, outputs_large], dim=-1)
        
        # get through the encoder
        output = self.encoder(combined_features)
        return output
    
    
''' ------------- some self-designed losses (criterion) ------------ '''

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


    
''' ----------- CLIP model ---------- '''
   
class CLIPModel(nn.Module):
    """
    The model of CLIP with temperature
    """
    
    def __init__(self, image_encoder, gene_encoder):
        super(CLIPModel, self).__init__()
        self.image_encoder = image_encoder
        self.gene_encoder = gene_encoder
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, img_small, img_large, gene_ids, gene_exp, mask):
        image_en = self.image_encoder(img_small, img_large)
        gene_en = self.gene_encoder(gene_ids, gene_exp, mask)
        return image_en, gene_en
    
def clip_loss(image_features, gene_features, temperature):
    '''
    the loss function for training of CLIP
    '''
    
    batch_size = image_features.shape[0]
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    gene_features = gene_features / gene_features.norm(dim=1, keepdim=True)
    
    # Compute logits
    logits = torch.matmul(image_features, gene_features.t()) / temperature
    
    # Create labels
    labels = torch.arange(batch_size).to(image_features.device)
    
    # Compute cross entropy loss
    loss_i2g = nn.CrossEntropyLoss()(logits, labels)
    loss_g2i = nn.CrossEntropyLoss()(logits.t(), labels)
    
    return (loss_i2g + loss_g2i) / 2
    


if __name__ == '__main__':
    pass