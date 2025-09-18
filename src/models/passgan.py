import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block for the generator and discriminator"""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 5, padding=2)
        self.conv2 = nn.Conv1d(dim, dim, 5, padding=2)

    def forward(self, x):
        residual = x
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return residual + 0.3 * x


class Generator(nn.Module):
    """Generator network for password generation"""

    def __init__(self, seq_len, layer_dim, output_dim, noise_dim=128):
        super().__init__()
        self.fc = nn.Linear(noise_dim, seq_len * layer_dim)
        self.blocks = nn.Sequential(
            ResBlock(layer_dim),
            ResBlock(layer_dim),
            ResBlock(layer_dim),
            ResBlock(layer_dim),
            ResBlock(layer_dim)
        )
        self.conv_out = nn.Conv1d(layer_dim, output_dim, 1)
        self.seq_len = seq_len
        self.layer_dim = layer_dim

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(-1, self.layer_dim, self.seq_len)
        x = self.blocks(x)
        x = self.conv_out(x)
        x = x.permute(0, 2, 1)  # [batch, seq, features]
        return torch.softmax(x, dim=2)


class Discriminator(nn.Module):
    """Discriminator network for password validation"""

    def __init__(self, seq_len, layer_dim, input_dim):
        super().__init__()
        self.conv_in = nn.Conv1d(input_dim, layer_dim, 1)
        self.blocks = nn.Sequential(
            ResBlock(layer_dim),
            ResBlock(layer_dim),
            ResBlock(layer_dim),
            ResBlock(layer_dim),
            ResBlock(layer_dim)
        )
        self.fc_out = nn.Linear(seq_len * layer_dim, 1)

    def forward(self, x):
        # Input shape: [batch, features, seq]
        x = self.conv_in(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_out(x)