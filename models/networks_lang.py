'''
Created on 24 May 2023

@author: yang
'''

import torch

from torch import nn
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer


class LLM_EmbedNet(nn.Module):
    def __init__(self, 
                 model_name='huawei-noah/TinyBERT_General_6L_768D', 
                 squeeze_dim=128,
                 text_max_len=1000):
        super(LLM_EmbedNet, self).__init__()
        
        self.name = 'LMEmbNet'
        self.longname = 'LMEmbNet_{}'.format(model_name)
        
        self.text_max_len = text_max_len
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if squeeze_dim <= 0 or squeeze_dim is None:
            self.squeeze = nn.Identity()
        else:
            self.squeeze = nn.Linear(self.transformer.config.hidden_size, squeeze_dim)

    def forward(self, text):
        encoded_input = self.tokenizer(text, truncation=True, max_length=self.text_max_len, 
                                       padding='max_length', return_tensors='pt')
        transformer_output = self.transformer(**encoded_input)
        pooled_output = transformer_output.pooler_output
        return self.squeeze(pooled_output)
    
class GPT_EmbedNet(LLM_EmbedNet):
    def __init__(self, 
                 model_name='distilgpt2', 
                 squeeze_dim=128,
                 text_max_len=1000):
        super(GPT_EmbedNet, self).__init__(model_name, squeeze_dim, text_max_len)

    def forward(self, text):
        encoded_input = self.tokenizer(text, truncation=True, max_length=self.text_max_len, 
                                       padding='max_length', return_tensors='pt')
        transformer_output = self.transformer(**encoded_input)
        pooled_output = transformer_output.last_hidden_state.mean(dim=1)
        return self.squeeze(pooled_output)

if __name__ == '__main__':
    pass