'''
Created on 2 Apr 2024

@author: laengs2304
'''
import platform

import torch


def _todevice(torch_item):
    devices = torch.device('cuda')
    devices_cpu = torch.device('cpu')
    devices_mps = torch.device('mps')
    
    OS_NAME = platform.system()
    
    if OS_NAME == 'Windows':
        if torch.cuda.is_available() == True:
            torch_item.to(devices)
        else:
            torch_item.to(devices_cpu)
    elif OS_NAME == 'Darwin': # mac
        if torch.backends.mps.is_available() == True:
            torch_item.to(devices_mps)
        else:
            torch_item.to(devices_cpu)
    else:
        torch_item.to(devices)

if __name__ == '__main__':
    pass