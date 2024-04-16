'''
Created on 16 Apr 2024

@author: huyang
'''

from support import env_st_pre
from trans import spot_tools


def test_1():
    ENV_task = env_st_pre.ENV_ST_HE_PRE
    trans_filename = 'DLPFC_151507_filtered_feature_bc_matrix.h5'
    bc_gene_dict, bcs, genes = spot_tools.parse_st_h5_f0_topvar0(ENV_task, trans_filename)
    print(list(bc_gene_dict.values())[0])
    
if __name__ == '__main__':
    test_1()