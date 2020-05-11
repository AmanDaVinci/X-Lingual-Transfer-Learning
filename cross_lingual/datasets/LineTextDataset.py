import torch
from typing import Dict, List, Tuple
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

class LineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: Path, block_size=512):
        assert file_path.is_file(), "Missing data file"
        
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

    def collate(self, examples: List[torch.Tensor], pad_token_id: int):
        return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)