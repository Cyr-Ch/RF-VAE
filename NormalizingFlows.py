import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class NormalizingFlows(nn.Module):
    def __init__(self, transforms, dim=2):

        super().__init__()
        if isinstance(transforms, nn.Module):
            self.transforms = nn.ModuleList([transforms, ])
        elif isinstance(transforms, list):
            if not all(isinstance(t, nn.Module) for t in transforms):
                raise ValueError("Wrong type of transforms")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(f"Wrong type of transforms")
        self.dim = dim
        self.base_dist = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def log_prob(self, x):

        inv_log_det = 0.0
        for transform in reversed(self.transforms):
            z, inv_log_det_jacobian = transform.inverse(x)
            inv_log_det += inv_log_det_jacobian
            x = z
        log_base = self.base_dist.log_prob(x)
        log_prob = (inv_log_det + log_base)

        return log_prob

    def sample(self, batch_size):

        x = self.base_dist.rsample([batch_size])
        log_base = self.base_dist.log_prob(x)
        log_det = 0.0
        for transform in self.transforms:
            x, log_det_jacobian = transform.forward(x)
            log_det += log_det_jacobian
        log_prob = - log_det + log_base

        return x, log_prob
