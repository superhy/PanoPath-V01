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
                                       nb_top_genes=1000,
                                       gene_n_heads=4,
                                       gene_n_layers=3,
                                       gene_dropout=0.25,
                                       gene_hidden_dim=128,
                                       block_gene_size=500,
                                       img_vit_patch_size=16,
                                       img_n_heads=4,
                                       img_n_layers=3,
                                       img_hidden_dim=128,
                                       clip_lr=1e-5,
                                       clip_batch_size=128, # have to be on multi-GPUs, otherwise <= 8
                                       clip_n_workers=12,
                                       clip_n_epochs=4000,
                                       grad_clip=False,
                                       clip_milestore=[0.01, 
                                                       0.05, 0.1, 0.15, 0.2, 0.25,
                                                       0.3, 0.35, 0.4, 0.45, 0.5, 
                                                       0.55, 0.6, 0.65, 0.7, 0.75,
                                                       0.8, 0.85, 0.9, 0.95, 1.0])

if __name__ == '__main__':
    pass