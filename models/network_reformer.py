'''
Created on 17 May 2024

@author: yanghu
'''


import torch
from transformers.models.reformer.configuration_reformer import ReformerConfig
from transformers.models.reformer.modeling_reformer import ReformerModel, \
    ReformerEmbeddings

import torch.nn as nn
import torch.nn.functional as F

''' ReFormer from HuggingFace '''

def pad_to_multiple_of_chunk_length(x, mask, chunk_length=64):
    '''
    Ensure the padding function matches the Reformer model's chunking requirements
    '''
    seq_length = x.shape[1]
    if seq_length % chunk_length != 0:
        padding_length = chunk_length - (seq_length % chunk_length)
        pad = (0, 0, 0, padding_length)
        x = F.pad(x, pad, "constant", 0)
        mask = F.pad(mask, pad, "constant", 1)
    return x, mask

class my_ReformerEmbeddings(ReformerEmbeddings):
    '''
    Custom Reformer Embeddings without positional embeddings
    '''
    def __init__(self, config):
        super().__init__(config)
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        # seq_length = input_shape[1]
        # if position_ids is None:
        #     position_ids = torch.arange(
        #         start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=torch.long, device=device
        #     )
        #     position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # if position_ids.shape[-1] > self.max_position_embeddings:
        #     raise ValueError(
        #         f"Sequence Length: {position_ids.shape[-1]} has to be less or equal than "
        #         f"config.max_position_embeddings {self.max_position_embeddings}."
        #     )

        # dropout
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        print(embeddings.shape)

        # add positional embeddings (disabled)
        return embeddings

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
        self.reformer.embeddings = my_ReformerEmbeddings(config)
        
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

        print(x.shape)
        # Ensure the sequence length is a multiple of chunk length (64)
        x, mask = pad_to_multiple_of_chunk_length(x, mask, chunk_length=64)
        print(x.shape, mask.shape)

        # Process through Reformer with mask support
        reformer_output = self.reformer(inputs_embeds=x, attention_mask=~mask).last_hidden_state
        print(reformer_output.shape)
        
        # Apply batch normalization and ReLU activation after reformer processing
        reformer_output = reformer_output.mean(dim=1)  # Aggregate across sequence dimension
        output = self.output_layer(reformer_output)
        output = self.norm(output)
        
        return output

if __name__ == '__main__':
    pass