"""
Server-side operations: gradient computation, local training, aggregation, and state tracking.
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE
from metrics import (flatten_grads, flatten_params, load_flat_params_,
                     probing_loss_random_offset)


def server_reference_grad(
    model: nn.Module, val_loader: DataLoader, batches: int = 10
) -> torch.Tensor:
    """
    Computes the server reference gradient by accumulating gradients
    over the validation set.

    Args:
        model:      global model
        val_loader: server validation dataloader
        batches:    number of batches to accumulate gradients over

    Returns:
        Flattened gradient tensor, shape (num_params,)
    """
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    for b, (x, y) in enumerate(val_loader):
        if b >= batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    gref = flatten_grads(model).detach()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    return gref


def local_train_delta(
    global_model: nn.Module,
    train_loader: DataLoader,
    lr: float = 0.01,
    steps: int = 10,
) -> torch.Tensor:
    """
    Computes the weight delta of a client by training a local copy of the
    global model for a fixed number of SGD steps.

    This function is used in the metrics phase to compute proj for all clients:
    the returned delta is projected onto the server gradient direction to measure
    client alignment.

    Args:
        global_model: global model to copy and train from
        train_loader: client training dataloader
        lr:           SGD learning rate
        steps:        number of SGD steps

    Returns:
        Weight delta (w_after - w_before), shape (num_params,)
    """
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    w0 = flatten_params(model).clone()

    it = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    w1 = flatten_params(model)
    return (w1 - w0).detach()


def compute_deltas_proj_mom_probe_now(
    model: nn.Module,
    client_train_loaders: List[DataLoader],
    client_eval_loaders: List[DataLoader],
    val_loader: DataLoader,
    local_lr: float,
    local_steps: int,
    probe_batches: int = 1,
    mom: Optional[torch.Tensor] = None,
    mom_beta: float = 0.90,
    round_seed: int = 0,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, torch.Tensor]:
    """
    Runs the metrics phase for all clients: computes weight deltas,
    proj and gener state metrics, and updates the server gradient EMA.

    For each client:
    - gener: probing loss of the global model on local data, evaluated at a
             random batch offset to reduce bias across rounds
    - proj:  dot product of the weight delta with the normalized server gradient EMA

    Args:
        model:                global model
        client_train_loaders: training dataloaders for all clients
        client_eval_loaders:  evaluation dataloaders for all clients
        val_loader:           server validation dataloader
        local_lr:             SGD learning rate
        local_steps:          number of SGD steps per client
        probe_batches:        number of batches used to compute gener
        mom:                  previous server gradient EMA (None for first round)
        mom_beta:             EMA coefficient for server gradient momentum
        round_seed:           base seed for reproducible random batch offset per client

    Returns:
        Tuple (deltas, proj, gener, mom) where:
        - deltas: list of weight deltas for all clients
        - proj:   projection scores onto EMA gradient direction, shape (N,)
        - gener:  generalization losses, shape (N,)
        - mom:    updated server gradient EMA
    """
    gref = server_reference_grad(model, val_loader, batches=10)

    if mom is None:
        mom = gref.detach().clone()
    else:
        mom = (mom_beta * mom) + ((1.0 - mom_beta) * gref.detach())

    desc_mom = (-mom).detach()
    desc_mom_norm = desc_mom / (desc_mom.norm() + 1e-12)

    deltas: List[torch.Tensor] = []
    probe_now: List[float] = []
    proj_mom: List[float] = []

    for i, (tr_loader, ev_loader) in enumerate(zip(client_train_loaders, client_eval_loaders)):
        rng_i = np.random.RandomState(int(round_seed) + 1000 + i)
        probe_now.append(
            float(probing_loss_random_offset(model, ev_loader, batches=probe_batches, rng=rng_i))
        )

        dw = local_train_delta(model, tr_loader, lr=local_lr, steps=local_steps)
        deltas.append(dw)

        proj_mom.append(float(torch.dot(dw, desc_mom_norm).item()))

    return (
        deltas,
        np.array(proj_mom, dtype=np.float32),
        np.array(probe_now, dtype=np.float32),
        mom.detach(),
    )


def apply_fedavg(
    model: nn.Module,
    deltas: List[torch.Tensor],
    selected: List[int],
) -> None:
    """
    Aggregates client deltas into the global model using simple coordinate-wise mean.

    Args:
        model:    global model to update in-place
        deltas:   list of weight delta tensors for all clients
        selected: indices of selected clients whose deltas will be averaged
    """
    w = flatten_params(model).clone()
    avg_dw = torch.stack([deltas[i] for i in selected], dim=0).mean(dim=0)
    load_flat_params_(model, w + avg_dw)


def update_staleness_streak(
    staleness: np.ndarray, streak: np.ndarray, selected: List[int]
) -> None:
    """
    Updates staleness and streak counters after each round.

    - staleness: incremented by 1 for non-selected clients, reset to 0 for selected
    - streak:    incremented by 1 for selected clients, reset to 0 for non-selected

    Args:
        staleness: staleness array, shape (N,), updated in-place
        streak:    streak array, shape (N,), updated in-place
        selected:  indices of clients selected in the current round
    """
    sel_mask = np.zeros(len(staleness), dtype=bool)
    sel_mask[selected] = True

    staleness[~sel_mask] += 1.0
    staleness[sel_mask] = 0.0

    streak[~sel_mask] = 0
    streak[sel_mask] += 1
