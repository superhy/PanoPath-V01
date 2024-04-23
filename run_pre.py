'''
Created on 22 Apr 2024

@author: super
'''
from support import env_st_pre
from trans.spot_process import _prod_st_cohort_names_from_folder, \
    _process_gen_spots_pkl_cohorts, _process_gen_spots_img_cohorts


# task_ids = [0.1]
task_ids = [1, 1.1]

ENV_task = env_st_pre.ENV_ST_HE_PRE

if __name__ == '__main__':
    
    if 0.1 in task_ids:
        '''
        load and generate the cohort file paths 
        '''
        _prod_st_cohort_names_from_folder(env_st_pre.ENV_ST_HE_PRE)
    if 1 in task_ids:
        mapping_csv_f_name = 'cohort_file_mapping.csv' 
        meta_csv_f_name = 'cohort_meta_info.csv'
        _process_gen_spots_pkl_cohorts(ENV_task, mapping_csv_f_name, meta_csv_f_name)
    if 1.1 in task_ids:
        mapping_csv_f_name = 'cohort_file_mapping.csv' 
        meta_csv_f_name = 'cohort_meta_info.csv'
        _process_gen_spots_img_cohorts(ENV_task, mapping_csv_f_name, meta_csv_f_name)