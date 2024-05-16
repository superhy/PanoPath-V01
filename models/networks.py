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


class ContextEfficientViT(nn.Module):
    """
    Vision Transformer using vit-pytorch for tissue image analysis with specified output dimension.
    """
    
    def __init__(self, image_size=224, patch_size=16, heads=4, depth=3, mlp_dim=256, output_dim=128):
        super(ContextEfficientViT, self).__init__()
        
        self.network_name = 'CtxEffViT'

        # Define the shared ViT model without a classification head
        self.vit_shared = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=0,  # This should be set to 0 if we only need features
            dim=output_dim,
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
            nn.Linear(2 * output_dim, output_dim),  # Using output_dim now for the combination
            nn.BatchNorm1d(output_dim),
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
    
    def __init__(self, hidden_dim, model_name='WinKawaks/vit-tiny-patch16-224'):
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
    
    def __init__(self, hidden_dim, model_name='WinKawaks/vit-tiny-patch16-224'):
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
    

def test_dual_stream_vit():
    model = ContextDualViT(hidden_dim=128)
    model.eval() 
    
    # random images
    img_small = torch.rand(1, 3, 224, 224)  
    img_large = torch.rand(1, 3, 224, 224)  

    output = model(img_small, img_large)
    print("Output feature vector:", output.shape)


if __name__ == '__main__':
    pass