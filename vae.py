import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dims=[100, 100]):

        super().__init__()
        self.latent_dim = latent_dim
        
        self.linear0 = nn.Linear(obs_dim, hidden_dims[0])
        self.linear1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear21 = nn.Linear(hidden_dims[1], latent_dim)
        self.linear22 = nn.Linear(hidden_dims[1], latent_dim)

        self.linear3 = nn.Linear(latent_dim, hidden_dims[0])
        self.linear31 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear4 = nn.Linear(hidden_dims[1], obs_dim)

    def encoder(self, x):

        h = torch.relu(self.linear1(torch.relu(self.linear0(x))))
        return self.linear21(h), self.linear22(h)

    def sample_with_reparam(self, mu, logsigma):

        sample = torch.empty_like(mu).normal_(0., 1.) * logsigma.exp() + mu
        return sample 

    def decoder(self, z):
        
        decode = torch.sigmoid(self.linear4(torch.relu(self.linear31(torch.relu(self.linear3(z))))))
        return decode


    def kl_divergence(self, mu, logsigma):
        
        kl_div = 0.5 * (mu.pow(2) + (2 * logsigma).exp() - 2 * logsigma - 1).sum(-1)
        return kl_div

    def elbo(self, x):

        mu, logsigma = self.encoder(x)
        z = self.sample_with_reparam(mu, logsigma)
        theta = self.decoder(z)
        log_obs_prob = -F.binary_cross_entropy(theta, x, reduction='none').sum(-1)
        kl = self.kl_divergence(mu, logsigma)
        elbo = log_obs_prob - kl
        return elbo

    def sample(self, num_samples):

        z = torch.empty(num_samples, self.latent_dim).normal_()
        theta = self.decoder(z)
        sample = torch.bernoulli(theta)
        return sample

