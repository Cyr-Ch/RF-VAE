import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import Dataset
import seaborn as sns



class MoonDataset(Dataset):
    def __init__(self, n_samples=1200, seed=0):
        self.n_samples = n_samples

        np.random.seed(seed)
        self.X, _ = make_moons(n_samples=n_samples, shuffle=True, noise=.05, random_state=None)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x


def density(model, loader=[], batch_size=100, mesh_size=5):

    with torch.no_grad():
        xx, yy = np.meshgrid(np.linspace(- mesh_size, mesh_size, num=batch_size), np.linspace(- mesh_size, mesh_size, num=batch_size))
        coords = np.stack((xx, yy), axis=2)
        coords_resh = coords.reshape([-1, 2])
        log_prob = np.zeros((batch_size**2))
        for i in range(0, batch_size**2, batch_size):
            data = torch.from_numpy(coords_resh[i:i+batch_size, :]).float()
            log_prob[i:i+batch_size] = model.log_prob(data).numpy()

        plt.scatter(x = coords_resh[:,0], y = coords_resh[:,1], c=np.exp(log_prob))
        for X in loader:
            plt.scatter(x = X[:,0], y = X[:,1], marker='x', c='white', alpha=.05)

        plt.show()

def visualize_mnist(data_loader, num_rows=2, num_cols=5):
    """
    Based on https://medium.com/@mrdatascience/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068
    """
    sns.set_style('white')
    num_total = num_rows * num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 2 * num_rows))
    for i in range(num_total):
        ax = axes[i // num_cols, i % num_cols]
        idx = np.random.randint(len(data_loader.dataset))
        x = data_loader.dataset[idx][0].view(28, 28).cpu().numpy()
        y = data_loader.dataset[idx][1]
        ax.imshow(x, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'Class: {y}')
    plt.tight_layout()
    plt.show()


def visualize_samples(samples, num_rows=4, num_cols=5):

    sns.set_style('white')
    num_total = num_rows * num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 2 * num_rows))
    for i in range(num_total):
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(samples[i], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'Sample #{i+1}')
    plt.tight_layout()
    plt.show()



