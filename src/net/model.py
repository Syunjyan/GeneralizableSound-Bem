import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np


class NeuPAT(nn.Module):
    def __init__(self, n_output_dims, config):
        encoding_config, network_config = (
            config["encoding_config"],
            config["network_config"],
        )
        super().__init__()
        self.model = tcnn.NetworkWithInputEncoding(
            encoding_config["n_dims"], n_output_dims, encoding_config, network_config
        )

    def forward(self, x):
        x = self.model(x).to(torch.float32).abs()
        return x


class NeuPAT_torch(nn.Module):
    def __init__(self, n_output_dims, config):
        encoding_config, network_config = (
            config["encoding_config"],
            config["network_config"],
        )
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(encoding_config["n_dims"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_output_dims),
        )

    def forward(self, x):
        x = self.model(x)
        return x
