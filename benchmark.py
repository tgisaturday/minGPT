import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import XLAStatsMonitor
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.model import GPT


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

class CharDataModule(LightningDataModule):

    def __init__(self, batch_size, num_workers, block_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_size = block_size

                                    
    def setup(self, stage=None):
        if not os.path.exists("input.txt"):
            os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

        # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
        text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
        self.train_dataset = CharDataset(text, self.block_size)  # one line of poem is roughly 50 characters

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    seed_everything(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--n_layer', default=22, type=int)
    parser.add_argument('--n_head', default=16, type=int)
    parser.add_argument('--n_embd', default=720, type=int)
    parser.add_argument('--learning_rate', default=6e-4, type=float)
    parser.add_argument('--block_size', default=128, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    dm = CharDataModule(args.batch_size, args.num_workers, args.block_size)
    dm.setup()
    model = GPT(
        vocab_size=dm.train_dataset.vocab_size,
        block_size=dm.train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate
    )

    lr_decay = LearningRateDecayCallback(
        learning_rate=6e-4,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(dm.train_dataset) * args.block_size
    )

    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=5,
        tpu_cores=8,
        gradient_clip_val=1.0,
        callbacks=[lr_decay, XLAStatsMonitor()],
    )
    trainer.fit(model, datamodule = dm )
