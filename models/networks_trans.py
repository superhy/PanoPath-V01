'''
Created on 12 Apr 2024

@author: yang hu
'''

import torch
from transformers import BertModel, BertConfig

import torch.nn as nn
from trans.spot_tools import GeneNameHashTokenizer


class GeneBasicTransformer(nn.Module):
    """
    The basic gene expression embedding transformer
    """
    
    def __init__(self, vocab_size, hidden_dim, n_heads, n_layers, dropout):
        super(GeneBasicTransformer, self).__init__()
        
        self.network_name = 'GeneBasicTransformer'
        
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.expr_embedding = nn.Linear(1, hidden_dim)
        self.combine_layer = nn.Linear(2 * hidden_dim, hidden_dim)  # Combine layer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads,
                                          num_encoder_layers=n_layers,
                                          num_decoder_layers=0,
                                          dropout=dropout)
        # self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, gene_ids, expr_values):
        gene_embeds = self.gene_embedding(gene_ids)
        expr_embeds = self.expr_embedding(expr_values.unsqueeze(-1))
        # print(gene_embeds.shape, expr_embeds.shape)
        
        # Combine embeddings by concatenation and then process
        x = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_layer(x)
        # Transformer without positional encoding
        x = self.transformer.encoder(x)
        # print(x.shape)
        # Some pooling or aggregation if necessary
        x = torch.mean(x, dim=1)
        
        return x
    
    
    
    
    
if __name__ == '__main__':
    pass