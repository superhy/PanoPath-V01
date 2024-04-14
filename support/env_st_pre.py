'''
Created on 3 Apr 2024

@author: super
'''
from support import parames
from support.env import ENV


ENV_ST_HE_PRE = parames.parame_st_task(project_name=ENV.PROJECT_NAME, 
                                       pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT, 
                                       tissue_stain='HE')

if __name__ == '__main__':
    pass