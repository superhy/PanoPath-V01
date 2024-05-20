'''
Created on 17 May 2024

@author: yang hu
'''
import os

import torch
from torch.utils.data.dataloader import DataLoader

from models import functions, datasets
from models.networks import CLIPModel, ContextViT
from models.networks_trans import GeneTransformer, BlockGeneTransformer
from support import env
from trans.spot_process import load_pyobject_from_pkl


def clip_training_spot(ENV_task, img_encoder, gene_encoder, 
                       nb_epochs=1000, multi_gpu=True,
                       milestons=[0.2, 0.4, 0.6, 0.8, 1.0]):
    '''
    function for training the clip on spatial transcriptomics spot, gene <-> image
    '''
    spot_pkl_dir = ENV_task.ST_HE_SPOT_PKL_FOLDER
    model_dir = ENV_task.ST_HE_MODEL_DIR
    
    # prepare the dataset
    dataset = datasets.SpotDataset(root_dir=spot_pkl_dir, 
                                   transform=functions.get_data_arg_transform())
    dataloader = DataLoader(dataset, batch_size=ENV_task.CLIP_BACTH_SIZE, 
                            shuffle=True, num_workers=ENV_task.CLIP_N_WORKERS, 
                            collate_fn=datasets.my_collate_fn,
                            pin_memory=True)
    
    # transfer items to device and prepare the CLIP model
    # img_encoder = env._todevice(img_encoder)
    # gene_encoder = env._todevice(gene_encoder)
    # clip_model = env._todevice(CLIPModel(img_encoder, gene_encoder))
    clip_model = CLIPModel(img_encoder, gene_encoder)
    
    # prepare the optimizer
    optimizer = functions.optimizer_adam_basic(clip_model)
    
    store_path = os.path.join(model_dir, 
                              f'CLIP_{img_encoder.network_name}_{img_encoder.network_name}.pth')
    
    print('>>> start to training')
    if multi_gpu == True:
        # functions.train_clip_multi_gpu(clip_model, dataloader, nb_epochs, optimizer, 
        #                                store_path=store_path,
        #                                milestons=milestons)
        functions.train_clip_multi_gpu_torch(clip_model, dataloader, nb_epochs, optimizer, 
                                             store_path=store_path,
                                             milestons=milestons)
    else:
        functions.train_clip(clip_model, dataloader, nb_epochs, optimizer, 
                             store_path=store_path,
                             milestons=milestons)


    
''' ----------- the running functions ----------- '''
    
def _run_clip_training_spot_test(ENV_task):
    
    gene_vocab_name = 'gene_tokenizer.pkl'
    tokenizer = load_pyobject_from_pkl(ENV_task.ST_HE_CACHE_DIR, gene_vocab_name)
    vocab_size = len(tokenizer.vocab)
    
    img_encoder = ContextViT(patch_size=ENV_task.IMG_VIT_PATCH_SIZE, 
                             heads=ENV_task.IMG_N_HEADS, 
                             depth=ENV_task.IMG_N_LAYERS, 
                             hidden_dim=ENV_task.IMG_HIDDEN_DIM)
    gene_encoder = BlockGeneTransformer(vocab_size, 
                                        n_heads=ENV_task.GENE_N_HEADS, 
                                        n_layers=ENV_task.GENE_N_LAYERS, 
                                        dropout=ENV_task.GENE_DROPOUT,
                                        block_size=ENV_task.BLOCK_GENE_SIZE,
                                        hidden_dim=ENV_task.GENE_HIDDEN_DIM)
    
    clip_training_spot(ENV_task, img_encoder, gene_encoder, 
                       nb_epochs=3, 
                       multi_gpu=True)
    
def _run_clip_training_spot_ivit_gblockt(ENV_task):
    
    gene_vocab_name = 'gene_tokenizer.pkl'
    tokenizer = load_pyobject_from_pkl(ENV_task.ST_HE_CACHE_DIR, gene_vocab_name)
    vocab_size = len(tokenizer.vocab)
    
    img_encoder = ContextViT(patch_size=ENV_task.IMG_VIT_PATCH_SIZE, 
                             heads=ENV_task.IMG_N_HEADS, 
                             depth=ENV_task.IMG_N_LAYERS, 
                             hidden_dim=ENV_task.IMG_HIDDEN_DIM)
    gene_encoder = BlockGeneTransformer(vocab_size, 
                                        n_heads=ENV_task.GENE_N_HEADS, 
                                        n_layers=ENV_task.GENE_N_LAYERS, 
                                        dropout=ENV_task.GENE_DROPOUT,
                                        block_size=ENV_task.BLOCK_GENE_SIZE,
                                        hidden_dim=ENV_task.GENE_HIDDEN_DIM)
    
    clip_training_spot(ENV_task, img_encoder, gene_encoder, 
                       nb_epochs=ENV_task.CLIP_N_EPOCHS, 
                       multi_gpu=True)

if __name__ == '__main__':
    pass


