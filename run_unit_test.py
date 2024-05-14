'''
Created on 16 Apr 2024

@author: huyang
'''

import os

import torch
from torch.utils.data.dataloader import DataLoader

from models import functions, datasets
from models.datasets import SpotDataset
from models.networks_trans import GeneBasicTransformer
from support import env_st_pre
from trans import spot_tools
from trans.spot_process import _h_analyze_ext_genes_for_all_barcodes, \
    load_file_names, get_coordinates_from_csv, get_barcode_from_coord_csv, \
    _h_statistic_spot_pkl_gene_feature, _h_count_spot_num, \
    load_pyobject_from_pkl


def test_spot_tools_1():
    '''
    test parse_st_h5_f0_topvar0
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    trans_filename = 'DLPFC_151508_filtered_feature_bc_matrix.h5'
    bc_gene_dict, bcs, genes = spot_tools.parse_st_h5_f0_topvar0(ENV_task, trans_filename)
    print(f'number of barcodes: {len(bcs)}')
    
    # print(bc_gene_dict[bcs[0]])
    # print(bc_gene_dict[bcs[1]])
    # print(bc_gene_dict[bcs[2]])
    
    for i in range(3):
        gene_idxs = [idx for idx, _ in bc_gene_dict[bcs[i]] ]
        gene_exp = [v for _, v in bc_gene_dict[bcs[i]] ]
        print(genes[gene_idxs])
        print(gene_exp)
        
def test_spot_process_1():
    '''
    test _h_analyze_ext_genes_for_all_barcodes
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    _h_analyze_ext_genes_for_all_barcodes(ENV_task)
    
def test_spot_process_2():
    '''
    test load_file_names
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    cohort_names = load_file_names(ENV_task, 'cohort_file_mapping.csv')
    print(cohort_names)
    
def test_spot_process_3():
    '''
    test get_coordinates_from_csv
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    corrd_csv_file_name = 'DLPFC_151508_projection.csv'
    barcodes = get_barcode_from_coord_csv(ENV_task, corrd_csv_file_name)
    print(f'number of barcodes: {len(barcodes)}')
    for barcode in barcodes:
        coord_x, coord_y = get_coordinates_from_csv(ENV_task, corrd_csv_file_name, barcode)
        print(coord_x, coord_y)
        
def test_spot_process_4():
    '''
    test _h_statistic_spot_pkl_gene_feature
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    _h_statistic_spot_pkl_gene_feature(ENV_task)
    
def test_spot_process_5():
    '''
    test _h_count_spot_num
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    _h_count_spot_num(ENV_task)
    
    
def test_embedding_gene_exp():
    '''
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    gene_vocab_name = 'gene_tokenizer.pkl'
    tokenizer = load_pyobject_from_pkl(ENV_task.ST_HE_CACHE_DIR, gene_vocab_name)
    vocab_size = len(tokenizer.vocab)  # size of vocab
    
    # initialize the Transformer
    model = GeneBasicTransformer(vocab_size, hidden_dim=512, n_heads=4, n_layers=3, dropout=0.3)
    
    test_spot_obj_name = os.path.join('CytAssist_11mm_FFPE_Human_Colorectal_Cancer', 
                                      'CytAssist_11mm_FFPE_Human_Colorectal_Cancer-AACAATCCGAGTGGAC-1.pkl')
    test_spot = load_pyobject_from_pkl(ENV_task.ST_HE_SPOT_PKL_FOLDER, test_spot_obj_name)
    
    # encode gene names
    # encoded_genes = tokenizer.encode(test_spot.gene_ids)
    encoded_genes_tensor = torch.tensor(test_spot.gene_ids, dtype=torch.long).unsqueeze(0) # add the dim for batch
    # trans the gene expression to tensor
    expr_values_tensor = torch.tensor(test_spot.gene_exp, dtype=torch.float).unsqueeze(0)
    
    print(encoded_genes_tensor)
    print(expr_values_tensor)
    
    encoded_vectors = model(encoded_genes_tensor, expr_values_tensor)
    print("Encoded vectors:", encoded_vectors)
    print("with shape:", encoded_vectors.shape)
    
def test_spot_dataloader():
    '''
    '''
    spot_pkl_dir = env_st_pre.ENV_ST_HE_PRE.ST_HE_SPOT_PKL_FOLDER
    dataset = SpotDataset(root_dir=spot_pkl_dir, transform=functions.get_data_arg_transform())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=datasets.my_collate_fn)
    
    for batch in data_loader:
        print(batch['img_small'].shape)  # Example access to the small image batch
        print(batch['gene_exp'].shape)  # Example access to the gene expression data
            
if __name__ == '__main__':
    
    # test_spot_tools_1()
    # test_spot_process_1()
    # test_spot_process_2()
    # test_spot_process_3()
    # test_spot_process_4()
    # test_spot_process_5()
    
    # test_embedding_gene_exp()
    
    test_spot_dataloader()
    
    
    