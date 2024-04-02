'''
Created on 24 May 2023

@author: yang
'''

from torch.utils.data.dataset import Dataset


class ClinicalLogs_Txt_Dataset(Dataset):
    
    '''
    '''
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __getitem__(self, idx):
        with open(self.filepaths[idx], 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def __len__(self):
        return len(self.filepaths)


if __name__ == '__main__':
    pass