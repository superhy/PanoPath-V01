'''
Created on 13 Apr 2024

@author: yang hu
'''
import collections
from contextlib import redirect_stdout
import gc
import os
from pathlib import Path
import pickle

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from support import env_st_pre
from trans import spot_tools
from wsi import slide_tools


def store_pyobject_to_pkl(store_folder,
                          store_object, store_pkl_name):
    if not os.path.exists(store_folder):
        os.makedirs(store_folder)
    with open(os.path.join(store_folder, store_pkl_name), 'wb') as f_pkl:
        pickle.dump(store_object, f_pkl)

def load_pyobject_from_pkl(store_folder,
                           object_pkl_name):
    pkl_filepath = os.path.join(store_folder, object_pkl_name)
    if Path(pkl_filepath).exists() is False:
        return None
    with open(pkl_filepath, 'rb') as f_pkl:
        store_pkl_obj = pickle.load(f_pkl)
    return store_pkl_obj


def find_file_with_prefix(prefix, folder_path):
    """
    Finds the file in the specified folder that starts with the given prefix.
    Assumes there is exactly one matching file per prefix.
    """
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            return filename
    return None

def _prod_st_cohort_names_from_folder(ENV_task):
    '''
    load all the cohort names from any data folder
    '''
    trans_folder_path = ENV_task.ST_HE_TRANS_FOLDER
    tissue_folder_path = ENV_task.ST_HE_TISSUE_FOLDER
    visium_folder_path = ENV_task.ST_HE_VISIUM_FOLDER
    coords_folder_path = ENV_task.ST_HE_COORDS_FOLDER
    suffix = "_filtered_feature_bc_matrix.h5"
        
    filenames = os.listdir(trans_folder_path)
    
    cohort_names = []
    # load all cohort names
    for filename in filenames:
        if filename.endswith(suffix):
            cohort_name = filename[:-len(suffix)]
            cohort_names.append(cohort_name)
            print(f'get cohort name: {cohort_name}')     
        
    # Lists to hold the data for each column in the DataFrame
    trans_files = []
    tissue_files = []
    visium_files = []
    coords_files = []
    # Loop through each cohort and find corresponding files in each folder
    for cohort in cohort_names:
        trans_files.append(find_file_with_prefix(cohort, trans_folder_path))
        tissue_files.append(find_file_with_prefix(cohort, tissue_folder_path))
        visium_files.append(find_file_with_prefix(cohort, visium_folder_path))
        coords_files.append(find_file_with_prefix(cohort, coords_folder_path))
    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'cohort': cohort_names,
        'trans': trans_files,
        'tissue': tissue_files,
        'visium': visium_files,
        'coords': coords_files
    })
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(ENV_task.ST_HE_META_DIR, 'cohort_file_mapping.csv'), index=False)
    
    return cohort_names

def load_file_names(ENV_task, mapping_csv_f_name):
    '''
    load the cohort_names
    '''
    csv_file_path = os.path.join(ENV_task.ST_HE_META_DIR, mapping_csv_f_name)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Find the row corresponding to the given cohort name
    return list(df['cohort'])

def query_file_names_for_cohort(ENV_task, mapping_csv_f_name, cohort_name):
    """
    Load data from a CSV file and return the file names for a given cohort name.

    Args:
    cohort_name (str): The name of the cohort to retrieve file names for.
    csv_file_path (str): Path to the CSV file containing the mapping.

    Returns:
    dict: A dictionary containing the file names for trans, tissue, visium, and coords.
    """
    csv_file_path = os.path.join(ENV_task.ST_HE_META_DIR, mapping_csv_f_name)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Find the row corresponding to the given cohort name
    row = df[df['cohort'] == cohort_name]
    # Check if the cohort name exists in the DataFrame
    if row.empty:
        print(f"No data found for cohort: {cohort_name}")
        return None
    
    # Extract the file names from the DataFrame
    file_names = {
        'cohort': row['cohort'].values[0], # cohort_id
        'trans': row['trans'].values[0],
        'tissue': row['tissue'].values[0],
        'visium': row['visium'].values[0],
        'coords': row['coords'].values[0]
    }
    
    return file_names

def query_file_meta_info_for_cohort(ENV_task, meta_csv_f_name, cohort_name):
    '''
    Load data from a CSV file and return the meta information for a given cohort name.

    Args:
    cohort_name (str): The name of the cohort to retrieve file names for.

    Returns:
    dict: A dictionary containing the file names for organ and spot_size.
    '''
    csv_file_path = os.path.join(ENV_task.ST_HE_META_DIR, meta_csv_f_name)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Find the row corresponding to the given cohort name
    row = df[df['cohort'] == cohort_name]
    # Check if the cohort name exists in the DataFrame
    if row.empty:
        print(f"No data found for cohort: {cohort_name}")
        return None
    
    # Extract the cohort metas from the DataFrame
    cohort_metas = {
        'cohort': row['cohort'].values[0], # cohort_id
        'organ': row['organ'].values[0],
        'spot_size': row['spot_size'].values[0],
    }
    
    return cohort_metas

def get_barcode_from_coord_csv(ENV_task, corrd_csv_file_name):
    '''
    this function is for test more likely
    laod all barcode in coord file of specific cohort
    '''
    csv_file_path = os.path.join(ENV_task.ST_HE_COORDS_FOLDER, corrd_csv_file_name)
    df = pd.read_csv(csv_file_path)
    return list(df['Barcode'])

def check_and_cleanup_cohort_folder(barcodes, folder_path):
    """
    Check if the number of files in a folder matches the number of barcodes.
    If they match, print that the process is complete. If not, delete all files in the folder.

    Args:
    barcodes (list): List of barcodes expected to match the number of files.
    folder_path (str): Path to the folder where the spot objects are stored.
    """
    flag_complete = False
    
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        num_files = len(files)

        if num_files >= len(barcodes):
            print("Processing of the cohort is complete.")
            flag_complete = True
        else:
            print(f"Expected {len(barcodes)} files, but found {num_files}. Cleaning up...")
            for file in files:
                os.remove(os.path.join(folder_path, file))
            print("All files have been removed for reprocessing.")

    except FileNotFoundError:
        print(f"No such directory: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return flag_complete


def get_coordinates_from_csv(ENV_task, corrd_csv_file_name, barcode):
    """
    Read a CSV file and return the X and Y coordinates for a given barcode.

    Args:
    csv_file_path (str): Path to the CSV file.
    barcode (str): The barcode to search for.

    Returns:
    tuple: A tuple containing the X and Y coordinates.
    """
    csv_file_path = os.path.join(ENV_task.ST_HE_COORDS_FOLDER, corrd_csv_file_name)
    df = pd.read_csv(csv_file_path)
    coordinates_row = df[df['Barcode'] == barcode]
    
    if not coordinates_row.empty:
        return coordinates_row['X Coordinate'].iloc[0], coordinates_row['Y Coordinate'].iloc[0]
    else:
        return None, None

def gen_spot_single_slide(ENV_task, cohort_n, file_names_dict, cohort_metas_dict):
    '''
    create and store the spot objects in 
    '''
    
    barcode_gene_dict, barcodes_1, all_gene_names = spot_tools.parse_st_h5_f0_topvar0(ENV_task,
                                                                             trans_filename=file_names_dict['trans'],
                                                                             top_n=ENV_task.NB_TOP_GENES)
    barcodes_2 = get_barcode_from_coord_csv(ENV_task, file_names_dict['coords'])
    set_1 = set(barcodes_1)
    set_2 = set(barcodes_2)
    barcodes = list(set_1.intersection(set_2))
    
    spot_pkl_folder = os.path.join(ENV_task.ST_HE_SPOT_PKL_FOLDER, cohort_n)
    spot_img_folder = os.path.join(ENV_task.ST_HE_SPOT_IMG_FOLDER, cohort_n)
    if not os.path.exists(spot_img_folder):
        os.makedirs(spot_img_folder)
        print(f'create new cohort folder at: {spot_img_folder}')
    if not os.path.exists(spot_pkl_folder):
        os.makedirs(spot_pkl_folder)
        print(f'create new cohort folder at: {spot_pkl_folder}')
    else:
        # check if the pkl generation is already done
        flag_complete = check_and_cleanup_cohort_folder(barcodes, spot_pkl_folder)
        if flag_complete:
            del barcode_gene_dict, barcodes, all_gene_names
            gc.collect()
            return
        
    for i, barcode in tqdm(enumerate(barcodes), total=len(barcodes), desc="Loading spot pkl"):
        tissue_filepath = os.path.join(ENV_task.ST_HE_TISSUE_FOLDER, file_names_dict['tissue'])
        coord_x, coord_y = get_coordinates_from_csv(ENV_task, 
                                                    corrd_csv_file_name=file_names_dict['coords'], 
                                                    barcode=barcode)
        if coord_x is None or coord_y is None:
            continue
        if barcode not in barcode_gene_dict.keys():
            continue
        
        gene_infos = barcode_gene_dict[barcode]
        gene_idx, gene_exp = zip(*gene_infos)
        spot_img_path = os.path.join(spot_img_folder, f'{cohort_n}-{barcode}.jpg')
        gene_names = spot_tools.load_gene_names_from_long_idx(all_gene_names, gene_idx)
        
        spot = spot_tools.Spot(cohort_name=cohort_n, barcode=barcode, cancer_type=cohort_metas_dict['organ'], 
                               spot_size=cohort_metas_dict['spot_size'], 
                               gene_names=gene_names, 
                               gene_exp=gene_exp, 
                               org_nb_gene=len(all_gene_names),  
                               slide_path=tissue_filepath, img_path=spot_img_path, 
                               coord_h=coord_y, coord_w=coord_x, 
                               small_h_s=None, small_h_e=None, small_w_s=None, small_w_e=None, 
                               large_h_s=None, large_h_e=None, large_w_s=None, large_w_e=None)
        spot_pkl_name = f'{cohort_n}-{barcode}.pkl'
        store_pyobject_to_pkl(spot_pkl_folder, spot, spot_pkl_name)
        
        # del gene_infos, gene_idx, gene_exp, gene_names, spot
    print(f'store {len(barcodes)} spot pkl files in folder: {spot_pkl_folder}')
        
    del barcode_gene_dict, barcodes, all_gene_names
    gc.collect()

def _process_gen_spots_pkl_cohorts(ENV_task, mapping_csv_f_name, meta_csv_f_name):
    '''
    load gene information and image store path for all spots from all cohorts
    only generate the spot object on disk, load image next time to save memory 
    '''
    # load cohort names (cohort_id)
    cohort_names = load_file_names(ENV_task, mapping_csv_f_name)
    for cohort_n in cohort_names:
        file_names_dict = query_file_names_for_cohort(ENV_task, mapping_csv_f_name, cohort_n)
        cohort_metas_dict = query_file_meta_info_for_cohort(ENV_task, meta_csv_f_name, cohort_n)
            
        gen_spot_single_slide(ENV_task, cohort_n, file_names_dict, cohort_metas_dict)
        
def _process_repair_spots_pkl_cohorts(ENV_task, mapping_csv_f_name, meta_csv_f_name, broken_cohort_names):
    '''
    repair to gene information and image store path for all spots from part cohorts (specified)
    only generate the spot object on disk, load image next time to save memory 
    '''
    # load cohort names (cohort_id)
    for cohort_n in broken_cohort_names:
        file_names_dict = query_file_names_for_cohort(ENV_task, mapping_csv_f_name, cohort_n)
        cohort_metas_dict = query_file_meta_info_for_cohort(ENV_task, meta_csv_f_name, cohort_n)
            
        gen_spot_single_slide(ENV_task, cohort_n, file_names_dict, cohort_metas_dict)
        
def save_img_single_spot(spot, tissue_img, spot_pkl_folder, spot_pkl_name):
    '''
    '''
    if tissue_img is None:
        if spot.slide_path.endswith('.jpg'):
            tissue_img = slide_tools.just_get_slide_from_normal(spot.slide_path)
            cropped_img, la_w_s, la_w_e, la_h_s, la_h_e = spot_tools.crop_spot_patch_from_img(tissue_img,
                                                                                              coord_x=spot.coord_w, coord_y=spot.coord_h,
                                                                                              patch_size=spot.spot_size)
        else:
            tissue_img = slide_tools.just_get_slide_from_openslide(spot.slide_path)
            cropped_img, la_w_s, la_w_e, la_h_s, la_h_e = spot_tools.crop_spot_patch_from_slide(tissue_img,
                                                                                        coord_x=spot.coord_w, coord_y=spot.coord_h,
                                                                                        patch_size=spot.spot_size)
    else:
        if spot.slide_path.endswith('.jpg'):
            cropped_img, la_w_s, la_w_e, la_h_s, la_h_e = spot_tools.crop_spot_patch_from_img(tissue_img,
                                                                                              coord_x=spot.coord_w, coord_y=spot.coord_h,
                                                                                              patch_size=spot.spot_size)
        else:
            cropped_img, la_w_s, la_w_e, la_h_s, la_h_e = spot_tools.crop_spot_patch_from_slide(tissue_img,
                                                                                        coord_x=spot.coord_w, coord_y=spot.coord_h,
                                                                                        patch_size=spot.spot_size)
    
    cropped_img.save(spot.img_path, "JPEG")
    # cropped_img.save(spot.img_path.replace('.jpg', '.png'), "PNG")
    # cropped_img.save(spot.img_path.replace('.png', '.jpg'), "JPEG")
    spot.reset_large_loc(la_w_s, la_w_e, la_h_s, la_h_e)
    store_pyobject_to_pkl(spot_pkl_folder, spot, spot_pkl_name)
    
    del spot
    gc.collect()
    return tissue_img
            
def _process_gen_spots_img_cohorts(ENV_task, mapping_csv_f_name):
    '''
    according the stored spot pkl to generate the spot pacth images
    '''
    cohort_names = load_file_names(ENV_task, mapping_csv_f_name)
    for cohort_n in cohort_names:
        file_names_dict = query_file_names_for_cohort(ENV_task, mapping_csv_f_name, cohort_n)
        barcodes = get_barcode_from_coord_csv(ENV_task, file_names_dict['coords'])
        spot_pkl_folder = os.path.join(ENV_task.ST_HE_SPOT_PKL_FOLDER, cohort_n)
        spot_img_folder = os.path.join(ENV_task.ST_HE_SPOT_IMG_FOLDER, cohort_n)
        
        tissue_img = None
        for i, barcode in tqdm(enumerate(barcodes), total=len(barcodes), desc="Loading spot image"):
            spot_pkl_name = f'{cohort_n}-{barcode}.pkl'
            spot = load_pyobject_from_pkl(spot_pkl_folder, spot_pkl_name)
            if spot is None:
                continue
            tissue_img = save_img_single_spot(spot, tissue_img, spot_pkl_folder, spot_pkl_name)
            
        print(f'extract {len(barcodes)} spot images in folder: {spot_img_folder}')
        
        del tissue_img
        gc.collect()
        
def _process_repair_spots_img_cohorts(ENV_task, mapping_csv_f_name, broken_cohort_names):
    '''
    according the stored spot pkl to repair/generate the spot pacth images, for specified cohorts
    '''
    for cohort_n in broken_cohort_names:
        file_names_dict = query_file_names_for_cohort(ENV_task, mapping_csv_f_name, cohort_n)
        barcodes = get_barcode_from_coord_csv(ENV_task, file_names_dict['coords'])
        spot_pkl_folder = os.path.join(ENV_task.ST_HE_SPOT_PKL_FOLDER, cohort_n)
        spot_img_folder = os.path.join(ENV_task.ST_HE_SPOT_IMG_FOLDER, cohort_n)
        
        tissue_img = None
        for i, barcode in tqdm(enumerate(barcodes), total=len(barcodes), desc="Loading spot image"):
            spot_pkl_name = f'{cohort_n}-{barcode}.pkl'
            spot = load_pyobject_from_pkl(spot_pkl_folder, spot_pkl_name)
            if spot is None:
                continue
            tissue_img = save_img_single_spot(spot, tissue_img, spot_pkl_folder, spot_pkl_name)
            
        print(f'extract {len(barcodes)} spot images in folder: {spot_img_folder}')
        
        del tissue_img
        gc.collect()
            

''' -------------------- some unit test functions --------------------- '''

def _h_batch_qualitify_info_in_h5_files(ENV_task):
    """
    Process all .h5 files in the given folder to collect dataset statistics.
    
    PS: the prefix '_h_' means recommend call at here
    
    Args:
        ENV_task: to get the folder_path (str), be like Path to the folder containing .h5 files.
    """
    folder_path = ENV_task.ST_HE_TRANS_FOLDER
    # load all filenames
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    barcode_sets = {}
    gene_name_sets = {}
    
    log_file_path = os.path.join(ENV_task.ST_HE_LOG_DIR, 'log-_h_batch_qualitify_info_in_h5_files.log')
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(log_file):
            # enumerate the file names
            for filename in tqdm(filenames, desc="Processing .h5 files"):
                filepath = os.path.join(folder_path, filename)
                cohort_name = filename.replace('_filtered_feature_bc_matrix.h5', '')
                
                # call get_matrix_from_h5 for single file
                count_matrix = spot_tools.get_matrix_from_h5(filepath)
                barcodes = count_matrix.barcodes
                all_gene_names = count_matrix.all_gene_names
                # statistic of barcodes in different will have repeat
                barcode_sets[cohort_name] = set(barcodes)
                # count gene names
                gene_name_sets[cohort_name] = set(all_gene_names)
            
            # check barcodes number for each cohort
            print("Barcodes count per cohort:")
            for cohort, bcs in barcode_sets.items():
                print(f"{cohort}: {len(bcs)} barcodes")
            
            # check if barcode is unique
            all_barcodes = set()
            for bcs in barcode_sets.values():
                all_barcodes.update(bcs)
            if len(all_barcodes) == sum(len(bcs) for bcs in barcode_sets.values()):
                print("All barcodes across cohorts are unique.")
            else:
                print(f'Unique barcodes number: {len(all_barcodes)}')
                print(f'Real barcode number: {sum(len(bcs) for bcs in barcode_sets.values()) }')
                print("There are overlapping barcodes across cohorts.")
            
            # check the repeat of gene names in different cohorts
            print("\nGene names count per cohort:")
            gene_name_matches = collections.defaultdict(list)
            for cohort, genes in gene_name_sets.items():
                print(f"{cohort}: {len(genes)} gene names")
                gene_name_matches[frozenset(genes)].append(cohort)
            
            # print the cohorts name with same list of gene names
            print("\nCohorts with identical gene names:")
            for genes, cohorts in gene_name_matches.items():
                if len(cohorts) > 1:
                    print(f"> [{', '.join(cohorts)}] have the same set of gene names.")
                else:
                    print(f"> {cohorts} has {len(genes)} gene names.")
    
    print(f'>>> Please find log output in: {log_file_path}.')
    
def _h_plot_h5_files_statistics(ENV_task):
    """
    Process all .h5 files in the given folder to collect dataset statistics
    and plot barcodes and gene names counts per cohort.
    
    PS: the prefix '_h_' means recommend call at here
    """
    folder_path = ENV_task.ST_HE_TRANS_FOLDER
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    barcode_counts = {}
    gene_name_counts = {}
    
    # Process each .h5 file
    for filename in tqdm(filenames, desc="Processing .h5 files"):
        filepath = os.path.join(folder_path, filename)
        cohort_name = filename.replace('_filtered_feature_bc_matrix.h5', '')
        
        # Load data using the previously defined function
        count_matrix = spot_tools.get_matrix_from_h5(filepath)
        barcodes = count_matrix.barcodes
        all_gene_names = count_matrix.all_gene_names
        
        # Collect statistics for barcodes and gene names
        barcode_counts[cohort_name] = len(set(barcodes))
        gene_name_counts[cohort_name] = len(set(all_gene_names))

    # Plotting the number of barcodes per cohort
    plt.figure(figsize=(12, 7))
    plt.bar(barcode_counts.keys(), barcode_counts.values(), color='skyblue')
    plt.xlabel('Cohort')
    plt.ylabel('Number of Barcodes')
    plt.title('Number of Barcodes per Cohort')
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    save_path_1 = os.path.join(ENV_task.ST_HE_LOG_DIR, 'Cohort-nb_barcode-dist.png')
    plt.savefig(save_path_1)
    print(f'save plot: {save_path_1}')
    # plt.show()

    # Plotting the number of gene names per cohort
    plt.figure(figsize=(12, 7))
    plt.bar(gene_name_counts.keys(), gene_name_counts.values(), color='salmon')
    plt.xlabel('Cohort')
    plt.ylabel('Number of Gene Names')
    plt.title('Number of Gene Names per Cohort')
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    save_path_2 = os.path.join(ENV_task.ST_HE_LOG_DIR, 'Cohort-nb_gene-dist.png')
    plt.savefig(save_path_2)
    print(f'save plot: {save_path_2}')
    # plt.show()
    
def _h_analyze_ext_genes_for_all_barcodes(ENV_task):
    """
    Analyze and plot the distribution of gene_values lengths for all .h5 files in the specified folder.
    """
    folder_path = ENV_task.ST_HE_TRANS_FOLDER
    lengths = []  # List to store the lengths of gene_values for each barcode
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    print("Processing .h5 files...")
    for filename in filenames:
        filepath = os.path.join(folder_path, filename)
        # cohort_name = filename.replace('_filtered_feature_bc_matrix.h5', '')
        
        # Process each file to get the barcode_gene_dict
        barcode_gene_dict, barcodes, _ = spot_tools.parse_st_h5_f0_topvar0(ENV_task, filepath)
        # Collect lengths of gene_values for each barcode and clean up memory
        for barcode in barcodes:
            gene_values = barcode_gene_dict[barcode]
            lengths.append(len(gene_values))
            
        # Optionally, to minimize memory use, clear the barcode_gene_dict
        del barcode_gene_dict
        gc.collect()
        
    np_gene_lengths = np.asarray(lengths)
    print(f'max length: {np.max(np_gene_lengths)}, min length: {np.min(np_gene_lengths)}, \
    average length: {np.average(np_gene_lengths)}')

    # Plotting the distribution of gene_values lengths
    # plt.figure(figsize=(10, 6))
    # # sns.displot(lengths, kde=True)
    # sns.histplot(lengths, kde=True, color='blue')
    # plt.title('Distribution of Extracted Genes Number Across All Cohorts')
    # plt.xlabel('Number of Extracted Genes')
    # plt.ylabel('Frequency')
    # plt.ylim(0, 200)
    # plt.tight_layout()
    # plt.show()
    
def _h_statistic_spot_pkl_gene_feature(ENV_task):
    '''
    Statistic how much is the average, max, min length of gene features we selected
    '''
    spot_pkl_folder_path = ENV_task.ST_HE_SPOT_PKL_FOLDER
    cohort_names = load_file_names(ENV_task, 'cohort_file_mapping.csv')
    
    cohort_gene_f_len_dict = {}
    for cohort_n in cohort_names:
        cohort_folder = os.path.join(spot_pkl_folder_path, cohort_n)
        filenames = [f for f in os.listdir(cohort_folder) if f.endswith('.pkl')]
        
        gene_f_len_list = []
        for pkl_name in tqdm(filenames, desc="Loading .pkl files"):
            spot = load_pyobject_from_pkl(cohort_folder, pkl_name)
            gene_f_len_list.append(len(spot.gene_exp))
            
        nd_gene_f_lens = np.array(gene_f_len_list)
        max_gene_f_len = np.max(nd_gene_f_lens)
        min_gene_f_len = np.min(nd_gene_f_lens)
        avg_gene_f_len = np.average(nd_gene_f_lens)
        print(f'In cohort: {cohort_n}, max_gene_f_length: {max_gene_f_len}, \
        min_gene_f_length: {min_gene_f_len}, avg_gene_f_length: {avg_gene_f_len}')
        
        cohort_gene_f_len_dict[cohort_n] = avg_gene_f_len
        
    data = pd.DataFrame(list(cohort_gene_f_len_dict.items()), columns=['Cohort', 'Value'])

    plt.figure(figsize=(10, 6))  
    sns.barplot(x='Cohort', y='Value', data=data, palette='viridis')
    plt.title('Gene Feature Length Across Cohorts')  
    plt.xlabel('Cohort')  
    plt.ylabel('Gene feature length')  
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_path_sta = os.path.join(ENV_task.ST_HE_LOG_DIR, 'Gene-feature-length.png')
    plt.savefig(save_path_sta)
    print(f'save plot: {save_path_sta}')
    # plt.show()
    
def _h_count_spot_num(ENV_task):
    '''
    count how many spot (patch images) we have
    '''
    spot_pkl_folder_path = ENV_task.ST_HE_SPOT_PKL_FOLDER
    cohort_names = load_file_names(ENV_task, 'cohort_file_mapping.csv')
    
    nb_spot = 0
    for cohort_n in cohort_names:
        cohort_folder = os.path.join(spot_pkl_folder_path, cohort_n)
        filenames = [f for f in os.listdir(cohort_folder) if f.endswith('.pkl')]
        nb_spot += len(filenames)
    print(f'totally have {nb_spot} spots.')

if __name__ == '__main__':
    '''
    some unit tests here
    '''
    # _prod_st_cohort_names_from_folder(env_st_pre.ENV_ST_HE_PRE)
    # c_file_names = query_file_names_for_cohort(env_st_pre.ENV_ST_HE_PRE,
    #                                            'cohort_file_mapping.csv',
    #                                            'CytAssist_11mm_FFPE_Human_Kidney')
    # print(c_file_names)
    
    # _h_batch_qualitify_info_in_h5_files(env_st_pre.ENV_ST_HE_PRE)
    # _h_plot_h5_files_statistics(env_st_pre.ENV_ST_HE_PRE)
    _h_analyze_ext_genes_for_all_barcodes(env_st_pre.ENV_ST_HE_PRE)
    
    
    