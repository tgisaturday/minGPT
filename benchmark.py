import logging
import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset, DataLoader

from mingpt.callback import CUDACallback
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


if __name__ == '__main__':
    seed_everything(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--n_layer', default=22, type=int)
    parser.add_argument('--n_head', default=16, type=int)
    parser.add_argument('--n_embd', default=3072, type=int)
    parser.add_argument('--learning_rate', default=6e-4, type=float)
    parser.add_argument('--block_size', default=128, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, args.block_size)  # one line of poem is roughly 50 characters
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate
    )

    lr_decay = LearningRateDecayCallback(
        learning_rate=6e-4,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * args.block_size
    )
    config = {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.998, 0.999],
                "eps": 1e-5,
                "weight_decay": 1e-9,
            },
        },
        'scheduler': {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            }
        },
        "zero_optimization": {
            "stage": 3,
            "cpu_offload": True,
            "cpu_offload_params": True,
            "cpu_offload_use_pin_memory": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "allgather_partitions": True,
        }
    }

    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=1,
        gradient_clip_val=1.0,
        plugins=DeepSpeedPlugin(config=config, logging_level=logging.INFO),
        checkpoint_callback=False,
        callbacks=[lr_decay, CUDACallback()],
    )
    trainer.fit(model, train_loader)
