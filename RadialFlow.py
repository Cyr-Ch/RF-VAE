import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseRadial(nn.Module):
    # Radial Transformation

    def __init__(self, dim=2):

        super().__init__()
        self.dim = dim
        self.x0 = nn.Parameter(torch.Tensor(self.dim, ))
        self.pre_alpha = nn.Parameter(torch.Tensor(1, ))
        self.pre_beta = nn.Parameter(torch.Tensor(1, ))

        stdv = 1. / np.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)


    def inverse(self, x):

        alpha = F.softplus(self.pre_alpha)
        beta = -alpha + F.softplus(self.pre_beta)
        diff = x - self.x0
        r = diff.norm(dim=-1, keepdim=True)
        h = 1. / (alpha + r)
        y = x + beta * h * diff
        h_prime = - (h ** 2)
        l1 = beta * h
        l2 = beta * h_prime
        log_det_jac = ((self.dim - 1) * torch.log1p(l1) + torch.log1p(l1 + l2 * r)).sum(-1)

        return y, log_det_jac

    def forward(self, y):

        raise ValueError("There is no closed form.")


