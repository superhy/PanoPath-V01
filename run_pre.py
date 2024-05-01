'''
Created on 22 Apr 2024

@author: super
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

from run_main import Logger
from support import env_st_pre, tools
from trans.spot_process import _prod_st_cohort_names_from_folder, \
    _process_gen_spots_pkl_cohorts, _process_gen_spots_img_cohorts, \
    _process_repair_spots_pkl_cohorts, _process_repair_spots_img_cohorts, \
    load_all_gene_names_vocab


# task_ids = [0.9]
task_ids = [1]
# task_ids = [1.1]
# task_ids = [1.3]

ENV_task = env_st_pre.ENV_ST_HE_PRE

task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])

if __name__ == '__main__':
    
    log_name = 'pre_log-{}-{}.log'.format(task_str,
                                          str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.ST_HE_LOG_DIR, log_name))
    
    if 0.1 in task_ids:
        '''
        load and generate the cohort file paths 
        '''
        _prod_st_cohort_names_from_folder(env_st_pre.ENV_ST_HE_PRE)
    if 0.9 in task_ids:
        mapping_csv_f_name = 'cohort_file_mapping.csv' 
        load_all_gene_names_vocab(ENV_task, mapping_csv_f_name)
    if 1 in task_ids:
        mapping_csv_f_name = 'cohort_file_mapping.csv' 
        meta_csv_f_name = 'cohort_meta_info.csv'
        gene_vocab_name = 'gene_tokenizer.pkl'
        _process_gen_spots_pkl_cohorts(ENV_task, mapping_csv_f_name, meta_csv_f_name, gene_vocab_name)
    if 1.1 in task_ids:
        '''
        repair part cohorts' pkl
        '''
        mapping_csv_f_name = 'cohort_file_mapping.csv' 
        meta_csv_f_name = 'cohort_meta_info.csv'
        gene_vocab_name = 'gene_tokenizer.pkl'
        broken_cohort_names = []
        _process_repair_spots_pkl_cohorts(ENV_task, mapping_csv_f_name, meta_csv_f_name, 
                                          broken_cohort_names, gene_vocab_name)
    if 1.2 in task_ids:
        mapping_csv_f_name = 'cohort_file_mapping.csv' 
        _process_gen_spots_img_cohorts(ENV_task, mapping_csv_f_name)
    if 1.3 in task_ids:
        '''
        repair part cohorts' pkl
        '''
        mapping_csv_f_name = 'cohort_file_mapping.csv'
        broken_cohort_names = ['Visium_FFPE_Human_Cervical_Cancer',
                               'Visium_FFPE_Human_Intestinal_Cancer',
                               'Visium_FFPE_Human_Ovarian_Cancer',
                               'Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma',
                               'Visium_Human_Breast_Cancer'
                               ]
        _process_repair_spots_img_cohorts(ENV_task, mapping_csv_f_name, broken_cohort_names)
        
        
        