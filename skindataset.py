import pandas as pd
import torch
from torch.utils.data import Dataset

# import sys
# sys.path.insert(0, r"/root/autodl-tmp/Plant-Pathology/ResNet")
# sys.path.insert(0, r"/root/autodl-tmp/Plant-Pathology/Se-ResNext")

class SkinDataset(Dataset):
    """
    Author: Jun
    Usage : get dataset of skin_data.csv
    """

    def __init__(self, data_pave):
        self.data_pave = data_pave
        self.data = pd.read_csv(data_pave)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        beg_idx, end_idx = index*8, index*8 + 8

         # get sequence
        sequence = self.data.iloc[beg_idx:end_idx,0:3].values.astype(float)
        sequence = torch.tensor(sequence,dtype=torch.float32)

        # get label
        label = self.data.iloc[end_idx-1, 3]
        label = torch.tensor(label,dtype=torch.float32)
        return sequence, label

