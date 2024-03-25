import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(path):
    df_grf = pl.read_parquet(path  + "grf_random_l.parquet")
    df_grf_int = pl.read_parquet(path + "grf_random_l_int.parquet")
    
    n_samples = df_grf["group"].n_unique()
    
    df_grf = df_grf.filter(pl.col("x").is_in([round(x * 0.01, 2) for x in range(101)]))
    
    x = df_grf.filter(pl.col("group") == 0)["x"].to_numpy()
    y = df_grf_int.group_by("group", maintain_order=True).agg(pl.col("y"))["y"].explode().to_numpy().reshape(n_samples, -1)
    grfs = df_grf.group_by("group", maintain_order=True).agg(pl.col("grf"))["grf"].explode().to_numpy().reshape(n_samples, -1)  
    grf_ints = df_grf_int.group_by("group", maintain_order=True).agg(pl.col("grf_int"))["grf_int"].explode().to_numpy().reshape(n_samples, -1)

    y = y.astype(np.float32)
    grfs = grfs.astype(np.float32)
    grf_ints = grf_ints.astype(np.float32)

    return x, y, grfs, grf_ints, n_samples

def train_val_test_split(n_samples, grfs, y, grf_ints):
    n_train = int(0.8 * n_samples)  
    n_val = int(0.1 * n_samples)
    
    grf_train, grf_val, grf_test = grfs[:n_train], grfs[n_train:n_train + n_val], grfs[n_train + n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]  
    grf_int_train, grf_int_val, grf_int_test = grf_ints[:n_train], grf_ints[n_train:n_train + n_val], grf_ints[n_train + n_val:]
    
    return grf_train, grf_val, grf_test, y_train, y_val, y_test, grf_int_train, grf_int_val, grf_int_test

class IntegralData(Dataset):
    def __init__(self, grf, y, grf_int):
        self.grf = torch.tensor(grf)
        self.y = torch.tensor(y)
        self.grf_int = torch.tensor(grf_int)
        
    def __len__(self):
        return len(self.grf)
    
    def __getitem__(self, idx):
        return self.grf[idx], self.y[idx], self.grf_int[idx]
        
def collate_fn(batch):
    grf, y, grf_int = zip(*batch)
    grf = torch.stack(grf)
    y = torch.stack(y)
    grf_int = torch.stack(grf_int)
    return grf, y, grf_int