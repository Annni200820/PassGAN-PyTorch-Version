import collections
import numpy as np
import random
import os


def load_dataset(path, max_length, max_vocab_size=2048):
    """Load and preprocess the password dataset"""
    lines = []
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if len(line) > max_length:
                line = line[:max_length]
            lines.append(tuple(line) + (("`",) * (max_length - len(line))))

    random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)
    charmap = {'unk': 0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size - 1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            filtered_line.append(char if char in charmap else 'unk')
        filtered_lines.append(tuple(filtered_line))

    print(f"Loaded {len(lines)} lines")
    return filtered_lines, charmap, inv_charmap


class PasswordDataset:
    """Dataset class for password data"""

    def __init__(self, data, charmap):
        self.data = data
        self.charmap = charmap
        self.n_chars = len(charmap)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        indices = [self.charmap[c] for c in line]
        one_hot = np.zeros((len(line), self.n_chars), dtype=np.float32)
        one_hot[np.arange(len(line)), indices] = 1
        return one_hot