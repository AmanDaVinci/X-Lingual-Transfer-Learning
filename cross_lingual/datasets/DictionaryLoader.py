import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DictionaryDataset(Dataset):
    """Dictionary dataset."""

    def __init__(self, input_txt_file, transform=None, has_header=False):
        """
        Args:
            input_txt_file (string): Path to the txt dictionary
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if has_header is False:
            kwargs = {
                'header': None,
                'index_col': None
            }
        else:
            kwargs = {
                'header': 0,
                'index_col': 0
            }

        self.language_dict = pd.read_csv(input_txt_file, sep='\t', **kwargs)

        self.transform = transform

    def __len__(self):
        return self.language_dict.shape[0]

    def __getitem__(self, idx):
        sample = self.language_dict.iloc[idx].to_numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample


def merge_samples(batch):
    return np.stack(batch, axis=0)


if __name__ == '__main__':
    dataset = DictionaryDataset('data/filtered_en_nl_dict.txt')

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=merge_samples)

    for batch_idx, batch_samples in enumerate(dataloader):
        print(batch_idx, batch_samples)

    dataset = DictionaryDataset('data/dutch_top1k.txt')

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=merge_samples)

    for batch_idx, batch_samples in enumerate(dataloader):
        print(batch_idx, batch_samples)
