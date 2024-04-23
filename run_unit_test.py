'''
Created on 16 Apr 2024

@author: huyang
'''

from support import env_st_pre
from trans import spot_tools
from trans.spot_process import _h_analyze_ext_genes_for_all_barcodes, \
    load_file_names, get_coordinates_from_csv, get_barcode_from_coord_csv


def test_spot_tools_1():
    '''
    test parse_st_h5_f0_topvar0
    '''
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    trans_filename = 'DLPFC_151507_filtered_feature_bc_matrix.h5'
    bc_gene_dict, bcs, genes = spot_tools.parse_st_h5_f0_topvar0(ENV_task, trans_filename)
    
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
    corrd_csv_file_name = 'DLPFC_151507_projection.csv'
    barcodes = get_barcode_from_coord_csv(ENV_task, corrd_csv_file_name)
    for barcode in barcodes:
        coord_x, coord_y = get_coordinates_from_csv(ENV_task, corrd_csv_file_name, barcode)
        print(coord_x, coord_y)
            
if __name__ == '__main__':
    
    # test_spot_tools_1()
    # test_spot_process_1()
    test_spot_process_2()
    # test_spot_process_3()
    
    
    