"""Evaluate a trained checkpoint and print MAD / RMSE / MSE / MAPE.

Usage:
    python eval.py --config configs/default.yaml --ex Kimore_ex5 \
        --checkpoint checkpoints/Kimore_ex5.pt
"""
import torch

from config import load_config
from data import build_dataloaders
from engine import evaluate
from models import TemporalModel
from utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_loader, data_loader = build_dataloaders(
        ex=cfg["ex"],
        batch_size=cfg["batch_size"],
        test_size=cfg["test_size"],
        seed=cfg["split_seed"],
    )

    net = TemporalModel()
    net.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))

    metrics = evaluate(net, test_loader, data_loader=data_loader, device=device)
    print(f"\nResults for {cfg['ex']}  (checkpoint: {cfg['checkpoint']})")
    print(f"  MAD  : {metrics['mad']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MSE  : {metrics['mse']:.4f}")
    print(f"  MAPE : {metrics['mape']:.4f}")


if __name__ == "__main__":
    main()
