import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils import plot_density


def train(model, dataset, batch_size=150, max_epochs=1000, frequency=200):

    # Load dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Train model
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    for epoch in range(max_epochs + 1):
        total_loss = 0
        for batch_index, (X_train) in enumerate(train_loader):
            log_probs = model.log_prob(X_train)
            loss = -log_probs.mean(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        total_loss /= len(train_loader)
        losses.append(total_loss)

        if epoch % frequency == 0:
            print(f"Epoch {epoch} -> loss: {total_loss.item():.2f}")
            plot_density(model, train_loader)

    return model, losses