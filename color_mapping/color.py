import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class ColorDataset(Dataset):
    
    def __init__(self, color_data_path):
        self.color_data = np.load(color_data_path)
        self.orig_colors = self.color_data['extracted_colors_orig']
        self.proj_colors = self.color_data['extracted_colors_proj']

    def __len__(self):
        return len(self.orig_colors)
    
    def __getitem__(self, idx):
        return self.orig_colors[idx], self.proj_colors[idx]

class ColorTransform(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.color_transform = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return torch.clamp(self.color_transform(x), min=0, max=1)
    
    def auto_color_mapping(self, input):
        assert torch.all((input >= 0) & (input <= 1)), "Input values must be in range [0, 1]"
        original_shape = input.shape
        dims = list(original_shape)
        color_dim = len(dims) - 1 - dims[::-1].index(3)
        input_permuted = input.permute(*[i for i in range(len(dims)) if i != color_dim], color_dim)
        flat_shape = (-1, 3)
        input_reshaped = input_permuted.reshape(flat_shape)
        output = self.forward(input_reshaped)
        output = output.reshape(input_permuted.shape)
        inverse_permutation = [0] * len(dims)
        curr_idx = 0
        for i in range(len(dims)):
            if i != color_dim:
                inverse_permutation[i] = curr_idx
                curr_idx += 1
        inverse_permutation[color_dim] = len(dims) - 1
        return output.permute(*inverse_permutation)

color_mapping_path = os.path.abspath(os.path.dirname(__file__))

def load_color_mapper(model_path = os.path.join(color_mapping_path, 'epoch_543_valloss_0.000210.pth')):
    model = ColorTransform()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model