'''
Created on 13 Apr 2024

@author: yang hu
'''
import collections
import os

from tqdm import tqdm
from contextlib import redirect_stdout

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from support import env_st_pre
from trans import spot_tools
import gc


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

    # Plotting the distribution of gene_values lengths
    plt.figure(figsize=(10, 6))
    # sns.displot(lengths, kde=True)
    sns.histplot(lengths, kde=True, color='blue')
    plt.title('Distribution of Extracted Genes Number Across All Cohorts')
    plt.xlabel('Number of Extracted Genes')
    plt.ylabel('Frequency')
    plt.ylim(0, 200)
    plt.tight_layout()
    plt.show()

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
    
    
    