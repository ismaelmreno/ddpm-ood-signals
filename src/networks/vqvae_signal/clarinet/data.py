#####################################################################################
# MIT License                                                                       #
#                                                                                   #
# Copyright (C) 2018 Sungwon Kim                                                    #
#                                                                                   #
#   Permission is hereby granted, free of charge, to any person obtaining a copy    #
#   of this software and associated documentation files (the "Software"), to deal   #
#   in the Software without restriction, including without limitation the rights    #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
#   copies of the Software, and to permit persons to whom the Software is           #
#   furnished to do so, subject to the following conditions:                        #
#                                                                                   #
#   The above copyright notice and this permission notice shall be included in all  #
#   copies or substantial portions of the Software.                                 #
#                                                                                   #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
#   SOFTWARE.                                                                       #
#####################################################################################

import torch
from torch.utils.data import Dataset
import numpy as np
import os

use_cuda = torch.cuda.is_available()

max_time_steps = 6400
upsample_conditional_features = True
hop_length = 256


class LJspeechDataset(Dataset):
    def __init__(self, data_root, train=True, test_size=0.05):
        self.data_root = data_root
        self.lengths = []
        self.train = train
        self.test_size = test_size

        self.paths = [self.collect_files(0), self.collect_files(1)]

    def __len__(self):
        return len(self.paths[0])

    def __getitem__(self, idx):
        wav = np.load(self.paths[0][idx])
        mel = np.load(self.paths[1][idx])
        return wav, mel

    def interest_indices(self, paths):
        test_num_samples = int(self.test_size * len(paths))
        train_indices, test_indices = range(0, len(paths) - test_num_samples), \
            range(len(paths) - test_num_samples, len(paths))
        return train_indices if self.train else test_indices

    def collect_files(self, col):
        meta = os.path.join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths = list(map(lambda l: l.decode("utf-8").split("|")[col], lines))
        paths = list(map(lambda f: os.path.join(self.data_root, f), paths))

        # Filter by train/test
        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        self.lengths = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, self.lengths))
        return paths


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """
    Create batch

    Args : batch(tuple) : List of tuples / (x, c)  x : list of (T,) c : list of (T, D)

    Returns : Tuple of batch / Network inputs x (B, C, T), Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2

    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c = batch[idx]
            if upsample_conditional_features:
                assert len(x) % len(c) == 0 and len(x) // len(c) == hop_length

                max_steps = max_time_steps - max_time_steps % hop_length  # To ensure Divisibility

                if len(x) > max_steps:
                    max_time_frames = max_steps // hop_length
                    s = np.random.randint(0, len(c) - max_time_frames)
                    ts = s * hop_length
                    x = x[ts:ts + hop_length * max_time_frames]
                    c = c[s:s + max_time_frames]
                    assert len(x) % len(c) == 0 and len(x) // len(c) == hop_length
            else:
                pass
            new_batch.append((x, c))
        batch = new_batch
    else:
        pass

    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # x_batch : [B, T, 1]
    x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len) for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T')
        c_batch = torch.tensor(c_batch).transpose(1, 2).contiguous()
        del max_len
    else:
        c_batch = None

    # Convert to channel first i.e., (B, C, T) / C = 1
    x_batch = torch.tensor(x_batch).transpose(1, 2).contiguous()

    # Add extra axis i.e., (B, T, 1)
    y_batch = torch.tensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.tensor(input_lengths)
    return x_batch, y_batch, c_batch, input_lengths


def collate_fn_synthesize(batch):
    """
    Create batch

    Args : batch(tuple) : List of tuples / (x, c)  x : list of (T,) c : list of (T, D)

    Returns : Tuple of batch / Network inputs x (B, C, T), Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2

    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c = batch[idx]
            if upsample_conditional_features:
                assert len(x) % len(c) == 0 and len(x) // len(c) == hop_length
            new_batch.append((x, c))
        batch = new_batch
    else:
        pass

    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # x_batch : [B, T, 1]
    x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len) for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T')
        c_batch = torch.tensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    # Convert to channel first i.e., (B, C, T) / C = 1
    x_batch = torch.tensor(x_batch).transpose(1, 2).contiguous()

    # Add extra axis i.e., (B, T, 1)
    y_batch = torch.tensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.tensor(input_lengths)

    return x_batch, y_batch, c_batch, input_lengths
