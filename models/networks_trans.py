'''
Created on 12 Apr 2024

@author: yang hu
'''

import torch
from transformers.models.reformer.configuration_reformer import ReformerConfig
from transformers.models.reformer.modeling_reformer import ReformerModel

import torch.nn as nn


class GeneBasicTransformer(nn.Module):
    """
    The basic gene expression embedding transformer
    """
    
    def __init__(self, vocab_size, hidden_dim=128, n_heads=4, n_layers=3, dropout=0.2):
        super(GeneBasicTransformer, self).__init__()
        
        self.network_name = f'G_BasicT-h{n_heads}-d{n_layers}'
        
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.expr_embedding = nn.Linear(1, hidden_dim)
        self.combine_layer = nn.Linear(2 * hidden_dim, hidden_dim)  # Combine layer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads,
                                          num_encoder_layers=n_layers,
                                          num_decoder_layers=0,
                                          dropout=dropout,
                                          batch_first=True)
        # self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, gene_ids, expr_values, mask):
        gene_embeds = self.gene_embedding(gene_ids)
        expr_embeds = self.expr_embedding(expr_values.unsqueeze(-1))
        # print(gene_embeds.shape, expr_embeds.shape)
        
        # Combine embeddings by concatenation and then process
        x = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_layer(x)
        # take the mask to suit for inputs with different length
        x = x * mask.unsqueeze(-1)
        # Transformer without positional encoding
        x = self.transformer.encoder(x, src_key_padding_mask=mask) # apply the mask to transformer
        # print(x.shape)
        
        # Some pooling or aggregation if necessary
        x = torch.mean(x, dim=1) # masked part is not included in mean
        out = self.norm(x)
        
        return out
    

class GeneReformer(nn.Module):
    """
    The basic gene expression embedding transformer integrating a Reformer model
    """
    def __init__(self, vocab_size, 
                 hidden_dim, num_attention_heads=4, 
                 num_transformer_layers=3, 
                 dropout=0.2,
                 model_name = 'google/reformer-crime-and-punishment'):
        super(GeneReformer, self).__init__()
        
        model_str = model_name.replace('/', '_')
        self.network_name = f'GeneReformer_{model_str}'
        
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)  # Assuming padding idx is 0
        self.expr_embedding = nn.Linear(1, hidden_dim)
        self.combine_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # Load a pre-trained Reformer configured for not using positional embeddings
        config = ReformerConfig.from_pretrained(
            model_name,
            attention_head_size=hidden_dim // num_attention_heads,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_transformer_layers,
            feed_forward_size=hidden_dim * 2,
            hidden_dropout_prob=dropout,
            attention_dropout_prob=dropout,
            is_decoder=False,
            axial_pos_shape=(64, 64),  # Define but disable any axial positional encodings
            use_axial_pos_emb=False  # Ensure that no positional embeddings are used
        )
        self.reformer = ReformerModel(config)
        
        # Additional layer to project the output to the desired dimension
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, gene_ids, expr_values):
        gene_embeds = self.gene_embedding(gene_ids)
        expr_embeds = self.expr_embedding(expr_values.unsqueeze(-1))
        
        x = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_layer(x)

        # Process through Reformer, assume batch_first = True in configuration
        reformer_output = self.reformer(inputs_embeds=x).last_hidden_state
        
        # Apply batch normalization and ReLU activation after reformer processing
        reformer_output = reformer_output.mean(dim=1)  # Aggregate across sequence dimension
        output = self.output_layer(reformer_output)
        output = self.norm(output)
        
        return output
    
    
    
if __name__ == '__main__':
    pass