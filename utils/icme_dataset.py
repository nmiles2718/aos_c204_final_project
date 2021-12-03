import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

class ICMEDataset(Dataset):
    def __init__(self, icme_labels, rootdir, datadir, transform=None):
        """Pytorch dataset class for the ICME dataset

        Parameters
        ----------
        icme_labels : str
            CSV containing interval labels
        rootdir : str
            Base directory of the repo
        datadir : str
            Path to data directory
        transform : torch.transform
            Any pytorch transform we'd like to apply to the images
        """
        self.rootdir = rootdir
        self.datadir = datadir
        self.df = pd.read_csv(
            f'{rootdir}/data/{icme_labels}',
            header=0,
            parse_dates=['start_time', 'stop_time']
        )
        #         self.df['fname']
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = self.df.fname_img.iloc[idx]
        label = self.df.label.iloc[idx]
        img = np.load(f'{self.datadir}/{f}')
        return img, label