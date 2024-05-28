'''
@author: Yang Hu
'''
import os
import platform


class parames_basic():
    
    def __init__(self, 
                 project_name,
                 scale_factor=16,
                 tile_size=256,
                 transform_resize=224,
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
            self.PROJECT_DIR = os.path.join('/exafs1/well/rittscher/users/lec468/workspace',
                                            self.PROJECT_NAME)
            
        if self.OS_NAME == 'Windows':
            if os.environ.get('USERNAME') == 'laengs2304':
                self.DATA_DIR = 'D:/PanoPath-Project' # local
                # self.DATA_DIR = 'E:/PanoPath-Project' # SSD
            else:
                # self.DATA_DIR = 'E:/PanoPath-Project' # STAT
                self.DATA_DIR = 'F:/PanoPath-Project' # SSD
        elif self.OS_NAME == 'Darwin':
            self.DATA_DIR = '/Volumes/Extreme SSD/PanoPath-Project'
        else:
            self.DATA_DIR = '/exafs1/well/rittscher/users/lec468/PanoPath-Project' # on Linux servers
            
#         self.SLIDE_TYPE = slide_type
        self.SCALE_FACTOR = scale_factor
        self.PIL_IMAGE_FILE_FORMAT = pil_image_file_format
        self.TILE_H_SIZE = tile_size
        self.TILE_W_SIZE = self.TILE_H_SIZE
        self.TRANSFORMS_RESIZE = transform_resize
            
class parame_st_task(parames_basic):
    
    def __init__(self,
                 project_name,
                 scale_factor,
                 pil_image_file_format,
                 tissue_stain,
                 nb_top_genes,
                 gene_n_heads=4,
                 gene_n_layers=3,
                 gene_dropout=0.2,
                 gene_hidden_dim=128,
                 block_gene_size=500,
                 img_vit_patch_size=16,
                 img_n_heads=4,
                 img_n_layers=3,
                 img_hidden_dim=128,
                 clip_lr=1e-4,
                 clip_batch_size=4,
                 clip_n_workers=8,
                 clip_n_epochs=1000,
                 grad_clip=False,
                 clip_milestore=[0.2, 0.4, 0.6, 0.8, 1.0]
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
        self.ST_HE_MODEL_DIR = os.path.join(self.ST_HE_DIR, 'models')
        
        self.ST_IHC_TISSUE_FOLDER = os.path.join(self.ST_IHC_DIR, 'tissue')
        self.ST_IHC_TRANS_FOLDER = os.path.join(self.ST_IHC_DIR, 'trans')
        self.ST_IHC_VISIUM_FOLDER = os.path.join(self.ST_IHC_DIR, 'visium')
        
        self.TISSUE_STAIN = tissue_stain
        self.NB_TOP_GENES = nb_top_genes
        
        self.GENE_N_HEADS = gene_n_heads
        self.GENE_N_LAYERS = gene_n_layers
        self.GENE_DROPOUT = gene_dropout
        self.GENE_HIDDEN_DIM = gene_hidden_dim
        self.BLOCK_GENE_SIZE = block_gene_size
        self.IMG_VIT_PATCH_SIZE = img_vit_patch_size
        self.IMG_N_HEADS = img_n_heads
        self.IMG_N_LAYERS = img_n_layers
        self.IMG_HIDDEN_DIM = img_hidden_dim
        
        self.CLIP_LR = clip_lr
        self.CLIP_BACTH_SIZE = clip_batch_size 
        self.CLIP_N_WORKERS = clip_n_workers
        self.CLIP_N_EPOCHS = clip_n_epochs
        self.GRAD_CLIP = grad_clip
        self.CLIP_MILESTORE = clip_milestore
        

if __name__ == '__main__':
    pass