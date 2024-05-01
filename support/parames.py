'''
@author: Yang Hu
'''
import os
import platform


class parames_basic():
    
    def __init__(self, 
                 project_name,
                 scale_factor=16,
                 pil_image_file_format='.png'):
        """
        Args:
            project_name:, 
            project_dir: use project_name construct the project dir path,
            slide_type: dx or tx, default dx,
            scale_factor: scale ratio when visualization,
            tile_h_size: patch size to separate the whole slide image,
            tile_w_size,
            transforms_resize,
            tp_tiles_threshold,
            pil_image_file_format,
            debug_mode
        """
        
        self.OS_NAME = platform.system()
        self.PROJECT_NAME = project_name
        
        ''' some default dirs '''
        if self.OS_NAME == 'Windows':
            if os.environ.get('USERNAME') == 'laengs2304':
                self.PROJECT_DIR = os.path.join('D:/eclipse-workspace', self.PROJECT_NAME)
            else:
                self.PROJECT_DIR = os.path.join('D:/workspace', self.PROJECT_NAME)
        elif self.OS_NAME == 'Darwin':
            self.PROJECT_DIR = os.path.join('/Users/superhy/Documents/workspace', self.PROJECT_NAME)
        else:
            self.PROJECT_DIR = os.path.join('/home/cqj236/workspace', self.PROJECT_NAME)
            
        if self.OS_NAME == 'Windows':
            if os.environ.get('USERNAME') == 'laengs2304':
                # self.DATA_DIR = 'D:/PanoPath-Project' # local
                self.DATA_DIR = 'E:/PanoPath-Project' # SSD
            else:
                # self.DATA_DIR = 'E:/PanoPath-Project' # STAT
                self.DATA_DIR = 'F:/PanoPath-Project' # SSD
        elif self.OS_NAME == 'Darwin':
            self.DATA_DIR = '/Volumes/Extreme SSD/PanoPath-Project'
        else:
            self.DATA_DIR = '#TODO:' # on Linux servers
            
#         self.SLIDE_TYPE = slide_type
        self.SCALE_FACTOR = scale_factor
        self.PIL_IMAGE_FILE_FORMAT = pil_image_file_format
            
class parame_st_task(parames_basic):
    
    def __init__(self,
                 project_name,
                 scale_factor,
                 pil_image_file_format,
                 tissue_stain,
                 nb_top_genes,
                 ):
        
        super(parame_st_task, self).__init__(project_name, 
                                             scale_factor,
                                             pil_image_file_format)
        
        self.ST_HE_LOG_DIR = os.path.join(self.PROJECT_DIR, 'data/st-he/logs')
        self.ST_HE_META_DIR = os.path.join(self.PROJECT_DIR, 'data/st-he/meta')
        self.ST_HE_CACHE_DIR = os.path.join(self.PROJECT_DIR, 'data/st-he/cache')
        self.ST_HE_DIR = os.path.join(self.DATA_DIR, 'st-he')
        self.ST_IHC_DIR = os.path.join(self.DATA_DIR, 'st-ihc')
        
        self.ST_HE_COORDS_FOLDER = os.path.join(self.ST_HE_DIR, 'coords')
        self.ST_HE_TISSUE_FOLDER = os.path.join(self.ST_HE_DIR, 'tissue')
        self.ST_HE_TRANS_FOLDER = os.path.join(self.ST_HE_DIR, 'trans')
        self.ST_HE_VISIUM_FOLDER = os.path.join(self.ST_HE_DIR, 'visium')
        self.ST_HE_SPOT_IMG_FOLDER = os.path.join(self.ST_HE_DIR, 'spot_img')
        self.ST_HE_SPOT_PKL_FOLDER = os.path.join(self.ST_HE_DIR, 'spot_pkl')
        
        self.ST_IHC_TISSUE_FOLDER = os.path.join(self.ST_IHC_DIR, 'tissue')
        self.ST_IHC_TRANS_FOLDER = os.path.join(self.ST_IHC_DIR, 'trans')
        self.ST_IHC_VISIUM_FOLDER = os.path.join(self.ST_IHC_DIR, 'visium')
        
        self.TISSUE_STAIN = tissue_stain
        self.NB_TOP_GENES = nb_top_genes

if __name__ == '__main__':
    pass