import torch
import torch.nn as nn
import tinycudann as tcnn
from .dense_map import DenseMap
import numpy as np


class NeuralExpansionAT(nn.Module):
    def __init__(
        self,
        net_json_config,
        data_json_config,
        term_num=4,
    ):
        super().__init__()
        self.term_num = term_num
        map_config = net_json_config["dense_map"]
        self.map_num = map_config["map_num"]
        self.map_feat_dim = map_config["feats_dim"]
        self.dense_map = DenseMap(
            feat_dim=map_config["feats_dim"],
            resolution=map_config["resolution"],
            map_num=map_config["map_num"],
        )
        self.condition_encoding = tcnn.Encoding(
            net_json_config["condition_encoding"]["n_dims"],
            net_json_config["condition_encoding"],
        )
        self.freq_encoding = tcnn.Encoding(
            net_json_config["freq_encoding"]["n_dims"], net_json_config["freq_encoding"]
        )
        self.w_net = tcnn.Network(
            self.condition_encoding.n_output_dims,
            self.map_num,
            net_json_config["pos_net"],
        )
        self.map_net_in_dim = (
            self.freq_encoding.n_output_dims + self.dense_map.output_dim
        )
        self.map_net = tcnn.Network(
            self.map_net_in_dim,
            term_num,
            net_json_config["map_net"],
        )
        solver_config = data_json_config["solver"]
        r_max = torch.tensor(solver_config["r_max"], dtype=torch.float32)
        self.register_buffer("r_max", r_max)

    def forward(self, x):
        """
        x: torch.Tensor, shape [batch_size, 3(r, theta, phi) + condition_dim + 1(freq)], in range [0, 1]
        return: torch.Tensor, shape [batch_size, n_output_dims]
        """
        r = x[:, 0].unsqueeze(-1)
        theta = x[:, 1].unsqueeze(-1)
        phi = x[:, 2].unsqueeze(-1)
        condition_x = x[:, 3:-1]
        freq_x = x[:, -1].unsqueeze(-1)

        condition_encoded = self.condition_encoding(condition_x)
        map_weight = self.w_net(condition_encoded).float().abs() / self.map_num

        theta_phi = torch.cat([theta, phi], dim=-1)
        map_feats = self.dense_map(theta_phi).reshape(
            -1, self.map_num, self.map_feat_dim
        )
        freq_encoded = (
            self.freq_encoding(freq_x).unsqueeze(1).repeat(1, self.map_num, 1)
        )
        map_input = torch.cat([freq_encoded, map_feats], dim=-1)
        map_value = (
            self.map_net(map_input.reshape(-1, self.map_net_in_dim))
            .float()
            .abs()
            .reshape(-1, self.map_num, self.term_num)
        )

        r = r * (self.r_max - 1) + 1
        for i in range(self.term_num):
            map_value[:, :, i] *= 1.0 / (r ** (i + 1))
        map_value = map_value.sum(dim=-1)

        y = (map_value * map_weight).sum(dim=-1).unsqueeze(-1)
        return y
