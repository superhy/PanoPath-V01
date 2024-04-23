'''
Created on 3 Apr 2024

@author: super
'''
from support import parames
from support.env import ENV


ENV_ST_HE_PRE = parames.parame_st_task(project_name=ENV.PROJECT_NAME, 
                                       scale_factor=ENV.SCALE_FACTOR,
                                       pil_image_file_format=ENV.PIL_IMAGE_FILE_FORMAT, 
                                       tissue_stain='HE',
                                       nb_top_genes=1000)

if __name__ == '__main__':
    pass