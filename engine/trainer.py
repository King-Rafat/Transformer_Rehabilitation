"""Training loop."""
import os
import time

import torch
from tqdm import tqdm

from models import TemporalModel
from utils import performance_metrics


def _to_device(x, y, device):
    x, y = x.to(device), y.to(device)
    x = torch.unsqueeze(x, 1)
    b, d, t, n, c = x.size()
    x = x.view(-1, t, n, c)
    return x, y


def train(
    epochs,
    train_loader,
    test_loader,
    save_path,
    ex="model",
    lr=1e-4,
    weight_decay=0.0,
    huber_delta=0.1,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)

    net = TemporalModel().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)

    best_mad = float("inf")
    start = time.time()

    for epoch in range(epochs):
        net.train()
        preds, trues = [], []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs - 1}", ncols=70):
            x, y = _to_device(x, y, device)
            pred = net(x)
            loss = loss_func(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds.extend(pred.detach().flatten().cpu().numpy())
            trues.extend(y.detach().flatten().cpu().numpy())

        mad, _, _, _ = performance_metrics(trues, preds)

        # Validation
        val_mad = evaluate_mad(net, test_loader, device)
        print(f"[epoch {epoch}] train MAD={mad:.4f}  val MAD={val_mad:.4f}")

        if val_mad < best_mad:
            best_mad = val_mad
            ckpt = os.path.join(save_path, f"{ex}.pt")
            torch.save(net.state_dict(), ckpt)
            print(f"  saved best checkpoint -> {ckpt} (val MAD {best_mad:.4f})")

    print(f"Training done in {time.time() - start:.1f}s. Best val MAD={best_mad:.4f}")
    return net


@torch.no_grad()
def evaluate_mad(net, loader, device):
    net.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = _to_device(x, y, device)
        pred = net(x)
        preds.extend(pred.flatten().cpu().numpy())
        trues.extend(y.flatten().cpu().numpy())
    mad, _, _, _ = performance_metrics(trues, preds)
    return mad
