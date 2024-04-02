'''
@author: Yang Hu
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import pickle
import random
import sys

import numpy as np
from support.env import ENV
from support.files import clear_dir, parse_slide_caseid_from_filepath, \
    parse_slideid_from_filepath
from support.metadata import query_task_label_dict_fromcsv
from wsi import filter_tools
from wsi import slide_tools
from wsi import tiles_tools
# from support import env_monuseg, env_gtex_seg


sys.path.append("..")       
def generate_tiles_list_pkl_filepath(slide_filepath, tiles_list_pkl_dir):
    """
    generate the filepath of pickle 
    """
    
    slide_id = parse_slideid_from_filepath(slide_filepath)
    tiles_list_pkl_filename = slide_id + '-tiles.pkl'
    if not os.path.exists(tiles_list_pkl_dir):
        os.makedirs(tiles_list_pkl_dir)
    
    pkl_filepath = os.path.join(tiles_list_pkl_dir, tiles_list_pkl_filename)
    
    return pkl_filepath

 
def recovery_tiles_list_from_pkl(pkl_filepath):
    """
    load tiles list from [.pkl] file on disk
    (this function is for some other module)
    """
    with open(pkl_filepath, 'rb') as f_pkl:
        tiles_list = pickle.load(f_pkl)
    return tiles_list


def parse_filesystem_slide(slide_dir):
    slide_path_list = []
    for root, dirs, files in os.walk(slide_dir):
        for f in files:
            if f.endswith('.svs') or f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.ndpi'):
                slide_path = os.path.join(root, f)
                slide_path_list.append(slide_path)
                
    return slide_path_list

        
def slide_tiles_split_keep_object_u(ENV_task):
    """
    conduct the whole pipeline of slide's tiles split, by Sequential process
    store the tiles Object [.pkl] on disk
    
    without train/test separation, for [segmentation] and [un-supervised] task only  
    
    Args:
        slides_folder: the folder path of slides ready for segmentation
    """
    
    _env_slide_dir = ENV_task.SLIDE_FOLDER
    _env_tile_pkl_train_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR
    
    ''' load all slides '''
    slide_path_list = parse_filesystem_slide(_env_slide_dir)
    for i, slide_path in enumerate(slide_path_list):
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(slide_path)
        if ENV_task.STAIN_TYPE == 'PSR':
            np_small_filtered_img = filter_tools.apply_image_filters_psr(np_small_img)
        elif ENV_task.STAIN_TYPE == 'CD45':
            np_small_filtered_img = filter_tools.apply_image_filters_cd45(np_small_img)
        elif ENV_task.STAIN_TYPE == 'P62':
            np_small_filtered_img = filter_tools.apply_image_filters_p62(np_small_img)
        else:
            np_small_filtered_img = filter_tools.apply_image_filters_he(np_small_img)
        
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, slide_path,
                                                 ENV.TILE_W_SIZE, ENV.TILE_H_SIZE,
                                                 t_p_threshold=ENV.TP_TILES_THRESHOLD, load_small_tile=False)
        
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        pkl_path = generate_tiles_list_pkl_filepath(slide_path, _env_tile_pkl_train_dir)
        print('store the [.pkl] in {}'.format(pkl_path))
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
        
    
def slide_tiles_split_keep_object_cls(ENV_task, all_train=False):
    '''
    conduct the whole pipeline of slide's tiles split, by Sequential process
    store the tiles Object [.pkl] on disk
    
    with train/test separation, for classification task
    
    Args:
        ENV_task:
        test_num_set: list of number of test samples for 
        delete_old_files = True, this is a parameter be deprecated but setup as [True] default
    '''
    
    ''' preparing some file parames '''
    test_prop = ENV_task.TEST_PART_PROP
    _env_slide_dir = ENV_task.SLIDE_FOLDER
    _env_tile_pkl_train_dir = ENV_task.TASK_TILE_PKL_TRAIN_DIR
    _env_tile_pkl_test_dir = ENV_task.TASK_TILE_PKL_TEST_DIR
    _env_tp_tiles_threshold = ENV_task.TP_TILES_THRESHOLD
    _env_tile_w_size = ENV_task.TILE_W_SIZE
    _env_tile_h_size = ENV_task.TILE_H_SIZE
    
    slide_path_list = parse_filesystem_slide(_env_slide_dir)
    # [default do this], remove the old train and test pkl dir
    clear_dir([_env_tile_pkl_train_dir, _env_tile_pkl_test_dir])
    
    label_dict = query_task_label_dict_fromcsv(ENV_task)
    ''' cls_path_dict: {class: [](list of path), ...} '''
    cls_path_dict = {}
    for case_id in label_dict.keys():
        value = label_dict[case_id]
        if value not in cls_path_dict.keys():
            cls_path_dict[value] = []
    for slide_path in slide_path_list:
        case_id = parse_slide_caseid_from_filepath(slide_path)
        if case_id not in label_dict.keys():
            continue
        cls_path_dict[label_dict[case_id]].append(slide_path)
        
    # shuffle the slide_tiles_list for all classes
    for label_item in cls_path_dict.keys():
        cls_path_dict[label_item] = random.sample(cls_path_dict[label_item], len(cls_path_dict[label_item]) )
    # count the test(train) number for positive and negative samples
    cls_test_num_dict = {}
    for label_item in cls_path_dict.keys():
        cls_test_num_dict[label_item] = round(len(cls_path_dict[label_item]) * test_prop)
        
    print(cls_test_num_dict)
        
    # write train set
    print('<---------- store the train tiles list ---------->')
    train_allcls_path_list = []
    for label_item in cls_test_num_dict.keys():
        if all_train:
            train_allcls_path_list.extend(cls_path_dict[label_item][:] )
        else:
            train_allcls_path_list.extend(cls_path_dict[label_item][:-cls_test_num_dict[label_item]] )
    for train_slide_path in train_allcls_path_list:
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(train_slide_path)
        if ENV_task.STAIN_TYPE == 'PSR':
            np_small_filtered_img = filter_tools.apply_image_filters_psr(np_small_img)
        elif ENV_task.STAIN_TYPE == 'CD45':
            np_small_filtered_img = filter_tools.apply_image_filters_cd45(np_small_img)
        elif ENV_task.STAIN_TYPE == 'P62':
            np_small_filtered_img = filter_tools.apply_image_filters_p62(np_small_img)
        else:
            np_small_filtered_img = filter_tools.apply_image_filters_he(np_small_img)
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, train_slide_path,
                                                 _env_tile_w_size, _env_tile_h_size,
                                                 t_p_threshold=_env_tp_tiles_threshold, load_small_tile=False)
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (train_slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        pkl_path = generate_tiles_list_pkl_filepath(train_slide_path, _env_tile_pkl_train_dir)
        print('store the [.pkl] in {}'.format(pkl_path))
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
           
    # write test set
    print('<---------- store the test tiles list ---------->') 
    test_allcls_path_list = []
    for label_item in cls_test_num_dict.keys():
        if all_train:
            test_allcls_path_list.extend(cls_path_dict[label_item][:0] )
        else:
            test_allcls_path_list.extend(cls_path_dict[label_item][-cls_test_num_dict[label_item]:] )
    for test_slide_path in test_allcls_path_list:
        np_small_img, large_w, large_h, small_w, small_h = slide_tools.slide_to_scaled_np_image(test_slide_path)
        if ENV_task.STAIN_TYPE == 'PSR':
            np_small_filtered_img = filter_tools.apply_image_filters_psr(np_small_img)
        elif ENV_task.STAIN_TYPE == 'CD45':
            np_small_filtered_img = filter_tools.apply_image_filters_cd45(np_small_img)
        elif ENV_task.STAIN_TYPE == 'P62':
            np_small_filtered_img = filter_tools.apply_image_filters_p62(np_small_img)
        else:
            np_small_filtered_img = filter_tools.apply_image_filters_he(np_small_img)
        shape_set_img = (large_w, large_h, small_w, small_h)
        tiles_list = tiles_tools.get_slide_tiles(np_small_filtered_img, shape_set_img, test_slide_path,
                                                 _env_tile_w_size, _env_tile_h_size,
                                                 t_p_threshold=_env_tp_tiles_threshold, load_small_tile=False)
        print('generate tiles for slide: %s, keep [%d] tile objects in (.pkl) list.' % (test_slide_path, len(tiles_list)))
        if len(tiles_list) == 0:
            continue
        pkl_path = generate_tiles_list_pkl_filepath(test_slide_path, _env_tile_pkl_test_dir)
        print('store the [.pkl] in {}'.format(pkl_path))
        with open(pkl_path, 'wb') as f_pkl:
            pickle.dump(tiles_list, f_pkl)
        
            
def _run_monuseg_slide_tiles_split(ENV_task):
    '''
    '''
    slides_folder = os.path.join(ENV_task.SEG_TRAIN_FOLDER_PATH, 'images')
    _ = slide_tiles_split_keep_object_u(slides_folder)
    
def _run_gtexseg_slide_tiles_split(ENV_task):
    '''
    '''
    slides_folder = ENV_task.SEG_TRAIN_FOLDER_PATH
    _ = slide_tiles_split_keep_object_u(slides_folder)
    

if __name__ == '__main__': 
#     _run_monuseg_slide_tiles_split(env_monuseg.ENV_MONUSEG)
    # _run_gtexseg_slide_tiles_split(env_gtex_seg.ENV_GTEX_SEG) 
    
    pass
    
    
