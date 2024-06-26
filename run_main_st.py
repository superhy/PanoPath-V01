'''
Created on 24 Apr 2024

@author: yang hu
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

from run_main import Logger
from support import env_st_pre, tools
from models.functions_st import _run_clip_training_spot_test,\
    _run_clip_training_spot_ivit_gblockt


# task_ids = [0.1]
task_ids = [1.0]

ENV_task = env_st_pre.ENV_ST_HE_PRE

task_str = '-' + '-'.join([str(lbl) for lbl in task_ids])

if __name__ == '__main__':
    
    log_name = 'main_st_log-{}-{}.log'.format(task_str,
                                          str(tools.Time().start)[:13].replace(' ', '-'))
    sys.stdout = Logger(os.path.join(ENV_task.ST_HE_LOG_DIR, log_name))
    
    if 0.1 in task_ids:
        '''
        load and generate the cohort file paths 
        '''
        _run_clip_training_spot_test(ENV_task)
    if 1.0 in task_ids:
        _run_clip_training_spot_ivit_gblockt(ENV_task)
    