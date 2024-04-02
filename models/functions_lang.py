'''
Created on 24 May 2023

@author: yang
'''
from models.datasets_lang import ClinicalLogs_Txt_Dataset
from models import functions


def load_clilogs_embeddings(txt_filepaths, lm_embnet):
    '''
    '''
    txt_dataset = ClinicalLogs_Txt_Dataset(txt_filepaths)
    dataloader = functions.get_data_loader(txt_dataset, batch_size=16, num_workers=4, sf=False)
    
    for texts in dataloader:
        embeddings = lm_embnet(texts)


if __name__ == '__main__':
    pass