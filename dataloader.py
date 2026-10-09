import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.linalg import lstsq
import glob
import os


class load_matrix_data(Dataset):
    def __init__(self, x, y, cov_fea=None):

        self.x = x
        self.y = y
        self.cov_fea = cov_fea
        if self.cov_fea is not None:            
            assert self.cov_fea.shape[0] == self.y.shape[0]
            assert len(self.cov_fea.shape) == 2

    def __len__(self):
        return self.y.shape[0]
    
    def get_cls_num_list(self, num_bins, use_clump=True):
        min_value = torch.min(self.y)
        max_value = torch.max(self.y)

        hist = torch.histc(self.y, bins=num_bins, min=min_value, max=max_value)  # 100
        bin_edges = torch.linspace(min_value, max_value, steps=num_bins + 1)  # 101

        return hist, bin_edges


    def __getitem__(self, index):
        
        x = torch.FloatTensor(self.x[index, :]).unsqueeze(0)
        # x = (x - x.mean())/x.std()
        # x = (x - x.min())/(x.max() - x.min())
        x[torch.isnan(x)] = 0.0
        y = torch.FloatTensor(self.y[index])
        
        if self.cov_fea is not None:
            cov_fea = torch.FloatTensor(self.cov_fea[index, :]).unsqueeze(0)
            return x, y, cov_fea, index 
        else:
            return x, y, index


class load_matrix_data_MLP(Dataset):
    def __init__(self, x, y, cov_fea=None):

        self.x = x
        self.y = y
        self.cov_fea = cov_fea
        if self.cov_fea is not None:            
            assert self.cov_fea.shape[0] == self.y.shape[0]
            assert len(self.cov_fea.shape) == 2

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        
        x = torch.FloatTensor(self.x[index, :])
        # x = (x - x.mean())/x.std()
        # x = (x - x.min())/(x.max() - x.min())
        x[torch.isnan(x)] = 0.0
        y = torch.FloatTensor(self.y[index])
        
        if self.cov_fea is not None:
            cov_fea = torch.FloatTensor(self.cov_fea[index, :])
            return x, y, cov_fea, index 
        else:
            return x, y, index