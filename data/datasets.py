"""Dataset loading and DataLoader construction for KIMORE and UI-PRMD.

KIMORE uses the existing ``Data_Proc.data_processing.Data_Loader`` (which scales
inputs/targets). UI-PRMD is loaded from the raw CSV layout shipped with the
dataset. Both are returned as ready-to-use PyTorch ``DataLoader`` objects.
"""
import csv
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from Data_Proc.data_processing import Data_Loader


def load_uiprmd(path, root="./UI-PRMD", nr=80, n_dim=117, timesteps=240):
    """Load one UI-PRMD exercise folder into (correct/incorrect) arrays."""

    def _read(fname):
        with open(os.path.join(root, path, fname)) as f:
            return list(csv.reader(f))

    correct_x = np.asarray(_read("Data_Correct.csv"), dtype=float)
    correct_input = np.zeros((nr, timesteps, n_dim))
    for i in range(len(correct_x) // n_dim):
        correct_input[i] = np.transpose(correct_x[n_dim * i : n_dim * (i + 1), :])
    correct_label = np.asarray(_read("Labels_Correct.csv"), dtype=float)

    incorrect_x = np.asarray(_read("Data_Incorrect.csv"), dtype=float)
    incorrect_input = np.zeros((nr, timesteps, n_dim))
    for i in range(len(incorrect_x) // n_dim):
        incorrect_input[i] = np.transpose(incorrect_x[n_dim * i : n_dim * (i + 1), :])
    incorrect_label = np.asarray(_read("Labels_Incorrect.csv"), dtype=float)

    return correct_input, correct_label, incorrect_input, incorrect_label


def build_dataloaders(ex, batch_size=1, test_size=0.2, seed=420, nr=80):
    """Return ``(train_loader, test_loader, data_loader)``.

    ``data_loader`` is the KIMORE scaler object (or None for UI-PRMD); it is
    needed at eval time to invert the target scaling.
    """
    data_loader = None

    if "Kimore" in ex or "KIMORE" in ex:
        data_loader = Data_Loader(ex)
        train_x, test_x, train_y, test_y = train_test_split(
            data_loader.scaled_x,
            data_loader.scaled_y,
            test_size=test_size,
            random_state=seed,
        )
    else:
        c_x, c_y, i_x, i_y = load_uiprmd(ex, nr=nr)
        train_idx_c = np.random.choice(c_x.shape[0], int(nr * 0.8), replace=False)
        train_idx_i = np.random.choice(i_x.shape[0], int(nr * 0.8), replace=False)
        valid_idx_c = np.setdiff1d(np.arange(nr), train_idx_c)
        valid_idx_i = np.setdiff1d(np.arange(nr), train_idx_i)
        train_x = np.concatenate((c_x[train_idx_c], i_x[train_idx_i]))
        train_y = np.concatenate((np.squeeze(c_y[train_idx_c]), np.squeeze(i_y[train_idx_i])))
        test_x = np.concatenate((c_x[valid_idx_c], i_x[valid_idx_i]))
        test_y = np.concatenate((np.squeeze(c_y[valid_idx_c]), np.squeeze(i_y[valid_idx_i])))

    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)

    # UI-PRMD ships flat coordinates; expand the last axis to 3D points.
    if "UI" in ex:
        train_x = torch.unsqueeze(train_x, 3).expand(*train_x.shape, 3)
        train_y = torch.unsqueeze(train_y, 1)
        test_x = torch.unsqueeze(test_x, 3).expand(*test_x.shape, 3)
        test_y = torch.unsqueeze(test_y, 1)

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    test_ds = torch.utils.data.TensorDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, data_loader
