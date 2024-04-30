'''
Created on 12 Apr 2024

@author: yang hu
'''

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

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
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, gene_ids, expr_values):
        gene_embeds = self.gene_embedding(gene_ids)
        expr_embeds = self.expr_embedding(expr_values.unsqueeze(-1))
        
        # Combine embeddings by concatenation and then process
        x = torch.cat([gene_embeds, expr_embeds], dim=-1)
        x = self.combine_layer(x)
        # Transformer without positional encoding
        x = self.transformer.encoder(x)
        # Some pooling or aggregation if necessary
        x = torch.mean(x, dim=1)
        
        return self.output_layer(x)
    
    
class GeneNameHashTokenizer:
    """
    The basic Tokenizer function for gene name list
    just Hash each gene name to a id, from a big gene names vocab
    """
    
    def __init__(self, gene_names):
        
        self.network_name = 'GeneNameHashTokenizer'
        
        self.vocab = {gene: idx for idx, gene in enumerate(sorted(set(gene_names)))}
        self.inverse_vocab = {idx: gene for gene, idx in self.vocab.items()}

    def encode(self, gene_names):
        return [self.vocab.get(name, -1) for name in gene_names]

    def decode(self, indices):
        return [self.inverse_vocab.get(index, "<UNK>") for index in indices]
    
    
    
    
    
    
    
''' --------- some unit test functions --------- '''

def _test_embedding_gene_exp():
    '''
    '''
    all_gene_names = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6', 'gene7', 'gene8']
    tokenizer = GeneNameHashTokenizer(all_gene_names)
    vocab_size = len(tokenizer.vocab)  # size of vocab
    
    # initialize the Transformer
    model = GeneBasicTransformer(vocab_size, hidden_dim=512, n_heads=4, n_layers=3, dropout=0.3)
    
    sample_gene_names = ['gene1', 'gene4', 'gene6', 'gene2']
    sample_expr_values = [0.5, 1.2, 0.3, 0.7]
    
    # encode gene names
    encoded_genes = tokenizer.encode(sample_gene_names)
    encoded_genes_tensor = torch.tensor(encoded_genes, dtype=torch.long).unsqueeze(0) # add the dim for batch
    # trans the gene expression to tensor
    expr_values_tensor = torch.tensor(sample_expr_values, dtype=torch.float).unsqueeze(0)
    
    encoded_vectors = model(encoded_genes_tensor, expr_values_tensor)
    print("Encoded vectors:", encoded_vectors)

if __name__ == '__main__':
    pass