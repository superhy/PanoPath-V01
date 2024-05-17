'''
Created on 17 May 2024

@author: yang hu
'''
from torch.utils.data.dataloader import DataLoader

from models import functions, datasets
from models.networks import CLIPModel, ContextSmallViT
from models.networks_trans import GeneBasicTransformer
from support import env
from trans.spot_process import load_pyobject_from_pkl


def clip_training_spot(ENV_task, img_encoder, gene_encoder, nb_epochs=2000):
    '''
    function for training the clip on spatial transcriptomics spot, gene <-> image
    '''
    spot_pkl_dir = ENV_task.ST_HE_SPOT_PKL_FOLDER
    
    # prepare the dataset
    dataset = datasets.SpotDataset(root_dir=spot_pkl_dir, transform=functions.get_data_arg_transform())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, 
                             collate_fn=datasets.my_collate_fn,
                             pin_memory=True)
    
    # transfer items to device and prepare the CLIP model
    img_encoder = env._todevice(img_encoder)
    gene_encoder = env._todevice(gene_encoder)
    clip_model = env._todevice(CLIPModel(img_encoder, gene_encoder, hidden_dim=128))
    
    # prepare the optimizer
    optimizer = functions.optimizer_adam_basic(clip_model)
    
    functions.train_clip(clip_model, dataloader, nb_epochs, optimizer)


    
''' ----------- the running functions ----------- '''
    
def _run_clip_training_spot_test(ENV_task):
    
    gene_vocab_name = 'gene_tokenizer.pkl'
    tokenizer = load_pyobject_from_pkl(ENV_task.ST_HE_CACHE_DIR, gene_vocab_name)
    vocab_size = len(tokenizer.vocab)
    
    img_encoder = ContextSmallViT(hidden_dim=128)
    gene_encoder = GeneBasicTransformer(vocab_size=vocab_size, hidden_dim=128, n_heads=4, n_layers=3, dropout=0.3)

if __name__ == '__main__':
    pass