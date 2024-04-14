'''
Created on 13 Apr 2024

@author: yang hu
'''
import os

import pandas as pd
from torchvision.datasets import folder
from support import env_st_pre


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
        'trans': row['trans'].values[0],
        'tissue': row['tissue'].values[0],
        'visium': row['visium'].values[0],
        'coords': row['coords'].values[0]
    }
    
    return file_names
    
    

if __name__ == '__main__':
    '''
    some unit tests here
    '''
    # _prod_st_cohort_names_from_folder(env_st_pre.ENV_ST_HE_PRE)
    c_file_names = query_file_names_for_cohort(env_st_pre.ENV_ST_HE_PRE,
                                               'cohort_file_mapping.csv',
                                               'CytAssist_11mm_FFPE_Human_Kidney')
    print(c_file_names)
    
    
    