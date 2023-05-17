import torch
import numpy as np
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# ================================= PNA Aggregate Related Utils ==============================================

EPS = 1e-5

# Define all the aggregators
def aggregate_sum(h):
    return torch.sum(h, dim=1)
def aggregate_mean(h):
    return torch.mean(h, dim=1)
def aggregate_max(h):
    return torch.max(h, dim=1)[0]
def aggregate_min(h):
    return torch.min(h, dim=1)[0]
def aggregate_var(h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    return torch.relu(h_mean_squares - h_mean * h_mean)
def aggregate_std(h):
    return torch.sqrt(aggregate_var(h) + EPS)

# Define all the scalers
def scale_identity(h, D=None, log_degree=None):
    return h
def scale_amplification(h, D, log_degree):
    return h * (np.log(D + 1) / log_degree)
def scale_attenuation(h, D, log_degree):
    return h * (log_degree / np.log(D + 1))

# ================================= PNA Aggregate Related Utils ==============================================