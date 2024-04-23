'''
Created on 2 Apr 2024

@author: huyang
'''
import collections
import os

import anndata
import tables
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from support import env_st_pre
import gc


def get_matrix_from_h5(h5_filepath):
    '''
    load the matrix and full gene names from 10x-ST h5
    '''
    CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix', 'all_gene_names'])

    with tables.open_file(h5_filepath, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read().astype(str)
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
         
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read().astype(str)
        feature_names = getattr(feature_group, 'name').read().astype(str)
        feature_types = getattr(feature_group, 'feature_type').read().astype(str)
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types

        return CountMatrix(feature_ref, barcodes, matrix, feature_names)
    
def parse_st_h5_f0(ENV_task, trans_filename):
    '''
    load the key information from filtered_feature_bc_matrix, with .h5 anndata
    '''
    filtered_h5_path = os.path.join(ENV_task.ST_HE_TRANS_FOLDER, trans_filename)
    count_matrix = get_matrix_from_h5(filtered_h5_path)
    
    matrix = count_matrix.matrix
    barcodes = count_matrix.barcodes # Spot barcode
    gene_names = count_matrix.feature_ref['name'] # gene names
    
    # Convert the sparse matrix to dense format for easier processing
    dense_matrix = matrix.toarray().T
    print(f'with filtered matrix shape as (nb_spots, nb_genes): {dense_matrix.shape}')
    print(f'barcodes: \n{barcodes}')
    print(f'gene_names number: \n{len(gene_names)}')
    
    # Creating a dictionary with barcodes as keys and list of (gene_name, value) tuples as values
    barcode_gene_dict = {}
    for i, barcode in enumerate(barcodes):
        gene_infos = [(j, dense_matrix[i, j]) for j, _ in enumerate(gene_names) if dense_matrix[i, j] != 0]
        barcode_gene_dict[barcode] = gene_infos
    gc.collect() # release the memory
    
    return barcode_gene_dict, barcodes, gene_names

def parse_st_h5_topvar(ENV_task, trans_filename, top_n=1000):
    '''
    load the key information from filtered_feature_bc_matrix, including all and top variable genes.
    only keep the high-variable genes in the barcode_gene_dict
    '''
    filtered_h5_path = os.path.join(ENV_task.ST_HE_TRANS_FOLDER, trans_filename)
    count_matrix = get_matrix_from_h5(filtered_h5_path)
    
    matrix = count_matrix.matrix
    barcodes = count_matrix.barcodes
    all_gene_names = count_matrix.all_gene_names  # All gene_idx names
    
    # Compute variance for each gene_idx and filter top N variable genes
    variances = np.array(matrix.power(2).mean(axis=1) - np.square(matrix.mean(axis=1))).flatten()
    top_genes_indices = np.argsort(variances)[-top_n:]
    filtered_matrix = matrix[top_genes_indices, :]
    # top_var_gene_names = all_gene_names[top_genes_indices]  # Top variable gene_idx names
    
    # Convert the filtered sparse matrix to dense format for easier processing
    dense_matrix = filtered_matrix.toarray().T
    print(f'with filtered matrix shape as (nb_spots, nb_genes): {dense_matrix.shape}')
    print(f'barcodes: \n{barcodes}')
    print(f'top variable gene_idx names number: \n{len(top_genes_indices)}')
    
    # Creating a dictionary with barcodes as keys and list of (gene_name, value) tuples as values
    barcode_gene_dict = {}
    for i, barcode in enumerate(barcodes):
        gene_infos = [(gene_idx, dense_matrix[i, gene_idx]) for gene_idx in top_genes_indices]
        barcode_gene_dict[barcode] = gene_infos
    gc.collect() # release the memory
    
    return barcode_gene_dict, barcodes, all_gene_names

def parse_st_h5_f0_topvar0(ENV_task, trans_filename, top_n=1000):
    '''
    load the key information from filtered_feature_bc_matrix, with .h5 anndata
    keep non-zero gene (signal) + top variable zero gene (0)
    '''
    filtered_h5_path = os.path.join(ENV_task.ST_HE_TRANS_FOLDER, trans_filename)
    count_matrix = get_matrix_from_h5(filtered_h5_path)
    
    matrix = count_matrix.matrix
    barcodes = count_matrix.barcodes # Spot barcode
    # gene_names = count_matrix.feature_ref['name'] # gene names
    all_gene_names = count_matrix.all_gene_names
    if top_n is None:
        top_n = int(len(all_gene_names) / 4.0)
    
    # Convert the sparse matrix to dense format for easier processing
    dense_matrix = matrix.toarray().T
    print(f'with filtered matrix shape as (nb_spots, nb_genes): {dense_matrix.shape}')
    print(f'barcodes: \n{barcodes}')
    print(f'gene_names number: \n{len(all_gene_names)}')
    
    # Compute variance for each gene and filter top N variable genes
    variances = np.array(matrix.power(2).mean(axis=1) - np.square(matrix.mean(axis=1))).flatten()
    top_genes_indices = np.argsort(variances)[-top_n:]
    # filtered_matrix = matrix[top_genes_indices, :]
    # top_var_gene_names = all_gene_names[top_genes_indices]  # Top variable gene names
    
    # Creating a dictionary with barcodes as keys and list of (gene_name, value) tuples as values
    barcode_gene_dict = {}
    for i, barcode in tqdm(enumerate(barcodes), total=len(barcodes), desc="Processing gene matrix"):
        # gene_infos = [(gene, dense_matrix[i, j]) for j, gene in enumerate(all_gene_names) if dense_matrix[i, j] != 0]
        gene_infos = [(j, dense_matrix[i, j]) for j, _ in enumerate(all_gene_names) if dense_matrix[i, j] != 0]
        # nb_f0 = len(gene_infos)
        
        # gene_infos += [(top_var_gene_names[idx], 0) for idx in range(len(top_var_gene_names)) if dense_matrix[i, top_genes_indices[idx]] == 0]
        gene_infos += [(idx, 0) for idx in top_genes_indices if dense_matrix[i, idx] == 0]
        barcode_gene_dict[barcode] = gene_infos

        # print(f"Barcode: {barcode}, Original Non-Zero Genes: {nb_f0},\
        # Total after adding zeros: {len(gene_infos)},\
        # Added Zero-Expression Genes: {len(gene_infos) - nb_f0}")
    gc.collect() # release the memory
    
    return barcode_gene_dict, barcodes, all_gene_names

def load_gene_names_from_long_idx(all_gene_names, gene_idx):
    '''
    all_gene_names[gene_idx] will report "too many indices for array"
    use this function
    '''
    gene_names = []
    for gene_id in gene_idx:
        gene_names.append(all_gene_names[gene_id])
    return gene_names
    
def crop_spot_patch_from_slide(slide, coord_x, coord_y, patch_size):
    '''
    load the cropped patch for each Spot from slide file
    '''
    large_w_s = coord_x - patch_size // 2
    large_h_s = coord_y - patch_size // 2
    large_w_e = coord_x + patch_size // 2
    large_h_e = coord_y + patch_size // 2
    
    x, y = large_w_s, large_h_s
    w, h = large_w_e - large_w_s, large_h_e - large_h_s
    
    tile_region = slide.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    cropped_img = tile_region.convert("RGB")
    return cropped_img, large_w_s, large_w_e, large_h_s, large_h_e
    
def crop_spot_patch_from_img(img, coord_x, coord_y, patch_size):
    '''
    load the cropped patch for each Spot from normal image file (some slides are jpg)
    '''
    large_w_s = coord_x - patch_size // 2
    large_h_s = coord_y - patch_size // 2
    large_w_e = coord_x + patch_size // 2
    large_h_e = coord_y + patch_size // 2
    
    cropped_img = img.crop((large_w_s, large_w_e, large_h_s, large_h_e))
    return cropped_img, large_w_s, large_w_e, large_h_s, large_h_e

class Spot:
    """
    Class for information about a Spot of spatial transcriptomics
    In which: (h, w) -> (y, x) -> (r, c)
    
    Components:
    
    Functions:
    
    """
    
    def __init__(self, cohort_name, barcode, cancer_type, spot_size,
                 gene_names, gene_exp, org_nb_gene, 
                 slide_path, img_path, coord_h, coord_w, 
                 small_h_s, small_h_e, small_w_s, small_w_e, 
                 large_h_s, large_h_e, large_w_s, large_w_e):
        
        self.cohort_name = cohort_name
        self.barcode = barcode
        self.spot_id = f'{self.cohort_name}-{self.barcode}'
        self.cancer_type = cancer_type
        self.spot_size = spot_size
        self.gene_names = gene_names
        self.gene_exp = gene_exp
        self.ext_nb_gene = len(gene_names)
        self.org_nb_gene = org_nb_gene
        self.slide_path = slide_path
        self.img_path = img_path
        self.coord_h = coord_h
        self.coord_w = coord_w
        self.small_h_s = small_h_s
        self.small_h_e = small_h_e 
        self.small_w_s = small_w_s 
        self.small_w_e = small_w_e
        self.large_h_s = large_h_s
        self.large_h_e = large_h_e
        self.large_w_s = large_w_s
        self.large_w_e = large_w_e
        
    def reset_small_loc(self, sm_w_s, sm_w_e, sm_h_s, sm_h_e):
        self.small_h_s = sm_h_s
        self.small_h_e = sm_h_e 
        self.small_w_s = sm_w_s 
        self.small_w_e = sm_w_e
                
    def reset_large_loc(self, la_w_s, la_w_e, la_h_s, la_h_e):
        self.large_h_s = la_h_s
        self.large_h_e = la_h_e
        self.large_w_s = la_w_s
        self.large_w_e = la_w_e
    
    def __str__(self):
        return "[Cohort #%s, Barcode #%s, Number of Genes #%d, Small shape: (%d->%d, %d->%d), Large shape: (%d->%d, %d->%d)]" % (
          self.cohort_name, self.barcode, self.nb_gene, self.small_h_s, self.small_h_e,
          self.small_w_s, self.small_w_e, self.large_h_s, self.large_h_e, self.large_w_s, self.large_w_e)
    
    def __repr__(self):
        return "\n" + self.__str__()

if __name__ == '__main__':
    '''
    some unit tests here
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    trans_filename = 'DLPFC_151507_filtered_feature_bc_matrix.h5'
    # bc_gene_dict, bcs, genes = parse_st_h5_f0(ENV_task, trans_filename)
    # bc_gene_dict, bcs, genes = parse_st_h5_topvar(ENV_task, trans_filename)
    bc_gene_dict, bcs, genes = parse_st_h5_f0_topvar0(ENV_task, trans_filename)
    
    
    