'''
Created on 12 Apr 2024

@author: yang hu
'''

import torch

import torch.nn as nn

class GeneTransformer(nn.Module):
    """
    The basic gene expression embedding transformer
    """
    
    def __init__(self, vocab_size, n_heads=4, n_layers=3, dropout=0.2, hidden_dim=128):
        super(GeneTransformer, self).__init__()
        
        self.network_name = f'G-Tr-h{n_heads}-d{n_layers}'
        
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.expr_embedding = nn.Linear(1, hidden_dim)
        self.combine_layer = nn.Linear(2 * hidden_dim, hidden_dim)  # Combine layer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads,
                                          num_encoder_layers=n_layers,
                                          num_decoder_layers=0,
                                          dropout=dropout,
                                          batch_first=True)
        # self.output_layer = nn.Linear(hidden_dim, 1)
        
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
        out = torch.mean(x, dim=1) # masked part is not included in mean
        
        return out
    
class BlockGeneTransformer(nn.Module):
    """
    Gene expression embedding transformer with block-wise processing.
    """
    
    def __init__(self, vocab_size, n_heads=4, n_layers=3, dropout=0.2, block_size=512, hidden_dim=128):
        super(BlockGeneTransformer, self).__init__()
        
        self.network_name = f'G-Block_Tr-h{n_heads}-d{n_layers}'
        self.block_size = block_size
        
        self.gene_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.expr_embedding = nn.Linear(1, hidden_dim)
        self.combine_layer = nn.Linear(2 * hidden_dim, hidden_dim)  # Combine layer
        
        self.transformer_block = nn.Transformer(
            d_model=hidden_dim, nhead=n_heads, 
            num_encoder_layers=n_layers, num_decoder_layers=0, 
            dropout=dropout, batch_first=True
        )
        
    def check_tensor(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            print(f"Inf detected in {name}")
    
    def forward(self, gene_ids, expr_values, mask):
        gene_embeds = self.gene_embedding(gene_ids)
        expr_embeds = self.expr_embedding(expr_values.unsqueeze(-1))
        
        # Combine embeddings by concatenation and then process
        x = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_layer(x)
        
        # Divide x and mask into blocks
        num_blocks = (x.size(1) + self.block_size - 1) // self.block_size
        x_blocks = x.chunk(num_blocks, dim=1)
        mask_blocks = mask.chunk(num_blocks, dim=1)
        
        # Process each block separately
        block_outputs = []
        for i in range(num_blocks):
            x_block = x_blocks[i]
            mask_block = mask_blocks[i]
            
            # Transformer without positional encoding
            if not mask_block.any().item():
                block_output = self.transformer_block.encoder(x_block)
            else:
                block_output = self.transformer_block.encoder(x_block, 
                                                              src_key_padding_mask=mask_block) # apply the mask to transformer
            self.check_tensor(block_output, f"block_output_{i}")
            block_outputs.append(block_output)
        
        # Concatenate the block outputs
        x = torch.cat(block_outputs, dim=1)
        
        # Some pooling or aggregation if necessary
        out = torch.mean(x, dim=1)  # masked part is not included in mean
        # self.check_tensor(x, "mean_pooled_output")
        
        return out
    
    
if __name__ == '__main__':
    pass