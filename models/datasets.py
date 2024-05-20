'''
Created on 14 May 2024

@author: yang hu
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import pickle
import os
from models import functions


class SpotDataset(Dataset):
    """
    """
    
    def __init__(self, root_dir, transform=functions.get_data_arg_transform()):
        """
        Args:
            root_dir (string): Directory with all the spot pkl files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.spot_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.pkl')]
        self.transform = transform

    def __len__(self):
        return len(self.spot_files)

    def __getitem__(self, idx):
        # Load spot object from pkl file
        with open(self.spot_files[idx], 'rb') as file:
            spot = pickle.load(file)
        
        # Load images
        img_small = Image.open(spot.img_path).convert('RGB')
        img_large = Image.open(spot.context_img_path).convert('RGB')

        # Apply transforms to images if specified
        if self.transform:
            img_small = self.transform(img_small)
            img_large = self.transform(img_large)

        # Gene expression data (could be used as labels or inputs depending on your task)
        gene_ids = torch.tensor(spot.gene_ids, dtype=torch.int64)
        gene_exp = torch.tensor(spot.gene_exp, dtype=torch.float32)

        spot = {'img_small': img_small, 'img_large': img_large, 'gene_ids': gene_ids, 'gene_exp': gene_exp}

        return spot
    
def my_collate_fn(batch):
    '''
    '''
    
    # handle deal with the batch data
    batch_img_small = torch.stack([item['img_small'] for item in batch])
    batch_img_large = torch.stack([item['img_large'] for item in batch])
    
    # pad_sequence for gene sequence with different length, return the padding sequence and original length
    gene_ids = [item['gene_ids'].clone().detach() for item in batch]
    gene_exp = [item['gene_exp'].clone().detach() for item in batch]
    
    batch_gene_ids = pad_sequence(gene_ids, batch_first=True, padding_value=0)
    batch_gene_exp = pad_sequence(gene_exp, batch_first=True, padding_value=0.0)
    
    # create mask
    batch_mask = torch.zeros_like(batch_gene_ids, dtype=torch.bool)
    for i, length in enumerate([len(ids) for ids in gene_ids]):
        batch_mask[i, length:] = True
    # print(batch_mask)

    return {
        'img_small': batch_img_small,
        'img_large': batch_img_large,
        'gene_ids': batch_gene_ids,
        'gene_exp': batch_gene_exp,
        'mask': batch_mask  # return mask
    }

if __name__ == '__main__':
    pass