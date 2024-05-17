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
    
    
''' ReFormer from HuggingFace '''

def pad_to_multiple_of_chunk_length(x, mask, chunk_length=64):
    """
    Pads the input tensor to ensure its sequence length is a multiple of the chunk length.
    
    Args:
    x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_dim).
    mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length).
    chunk_length (int): The chunk length, default is 64.
    
    Returns:
    padded_x (torch.Tensor): The padded input tensor.
    padded_mask (torch.Tensor): The padded mask tensor.
    """
    seq_len = x.size(1)
    pad_len = (chunk_length - (seq_len % chunk_length)) % chunk_length
    
    if pad_len > 0:
        pad_tensor = torch.zeros((x.size(0), pad_len, x.size(2)), dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad_tensor], dim=1)
        
        pad_mask = torch.ones((mask.size(0), pad_len), dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask, pad_mask], dim=1)
    
    return x, mask

class GeneReformer(nn.Module):
    """
    The basic gene expression embedding transformer integrating a Reformer model
    """
    def __init__(self, vocab_size, 
                 hidden_dim, num_attention_heads=4, 
                 num_transformer_layers=3, 
                 dropout=0.2,
                 model_name='google/reformer-crime-and-punishment'):
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
            max_position_embeddings=16384  # Set this to a higher value if needed
        )
        
        self.reformer = ReformerModel(config)
        
        # Ensure position embeddings are set to None
        self.reformer.embeddings.position_embeddings = None # Disable position embeddings
        
        # Additional layer to project the output to the desired dimension
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, gene_ids, expr_values, mask):
        gene_embeds = self.gene_embedding(gene_ids)
        expr_embeds = self.expr_embedding(expr_values.unsqueeze(-1))
        
        x = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_layer(x)

        # Ensure the sequence length is a multiple of chunk length (64)
        x, mask = pad_to_multiple_of_chunk_length(x, mask, chunk_length=64)

        # Process through Reformer with mask support
        reformer_output = self.reformer(inputs_embeds=x, attention_mask=~mask).last_hidden_state
        
        # Apply batch normalization and ReLU activation after reformer processing
        reformer_output = reformer_output.mean(dim=1)  # Aggregate across sequence dimension
        output = self.output_layer(reformer_output)
        output = self.norm(output)
        
        return output
    
class my_ReformerEmbeddings(ReformerEmbeddings)
    
    
    
if __name__ == '__main__':
    pass