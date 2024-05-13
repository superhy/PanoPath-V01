'''
Created on 14 May 2024

@author: yang hu
'''


import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ContextShareViTransformer(nn.Module):
    """
    """
    
    def __init__(self, hidden_dim, model_name='google/vit-tiny-patch16-224'):
        super(ContextShareViTransformer, self).__init__()
        
        self.network_name = 'ContextShareViTransformer'
        
        # load the pre-trained ViT from HuggingFace
        config_img = ViTConfig.from_pretrained(model_name)
        config_img.num_labels = hidden_dim
        
        self.vit_shared = ViTModel(config_img)
        
        # using a layer to combine the features of small and large images
        self.encoder = nn.Linear(2 * config_img.hidden_size, hidden_dim)

    def forward(self, img_small, img_large):
        outputs_small = self.vit_shared(pixel_values=img_small).last_hidden_state[:, 0]
        outputs_large = self.vit_shared(pixel_values=img_large).last_hidden_state[:, 0]
        
        # combine the feature
        combined_features = torch.cat([outputs_small, outputs_large], dim=-1)
        
        # get through the encoder
        output = self.encoder(combined_features)
        return output

class ContextDualViTransformer(nn.Module):
    """
    """
    
    def __init__(self, hidden_dim, model_name='google/vit-tiny-patch16-224'):
        super(ContextDualViTransformer, self).__init__()
        
        self.network_name = 'ContextDualViTransformer'
        
        # load the pre-trained ViT from HuggingFace
        config_img = ViTConfig.from_pretrained(model_name)
        config_img.num_labels = hidden_dim
        
        self.vit_small = ViTModel(config_img)
        self.vit_large = ViTModel(config_img)
        
        # using a layer to combine the features of small and large images
        self.encoder = nn.Linear(2 * config_img.hidden_size, hidden_dim)

    def forward(self, img_small, img_large):
        outputs_small = self.vit_small(pixel_values=img_small).last_hidden_state[:, 0]
        outputs_large = self.vit_large(pixel_values=img_large).last_hidden_state[:, 0]
        
        # combine the feature
        combined_features = torch.cat([outputs_small, outputs_large], dim=-1)
        
        # get through the encoder
        output = self.encoder(combined_features)
        return output
    

def test_dual_stream_vit():
    model = ContextDualViTransformer(output_features=128)
    model.eval() 
    
    # random images
    img_small = torch.rand(1, 3, 224, 224)  
    img_large = torch.rand(1, 3, 224, 224)  

    output = model(img_small, img_large)
    print("Output feature vector:", output.shape)


if __name__ == '__main__':
    pass