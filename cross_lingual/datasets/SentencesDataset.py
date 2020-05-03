import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DictionaryDataset(Dataset):
    """Sentences dataset."""

    def __init__(self, input_txt_file, transform=None, has_header=False):
        """
        Args:
            input_txt_file (string): Path to the txt dictionary
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.samples = pd.read_csv(
            input_txt_file, sep='\n',
            index_col=None, header=None, squeeze=True)

        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def merge_samples(batch):
    return np.stack(batch, axis=0)


if __name__ == '__main__':
    dataset = DictionaryDataset('data/train.txt')

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=merge_samples)

    for batch_idx, batch_samples in enumerate(dataloader):
        print(batch_idx, batch_samples)
