"""Train the Point Cloud Transformer on a single exercise.

Usage:
    python train.py --config configs/default.yaml --ex Kimore_ex5 --epoch 2000
"""
from config import load_config
from data import build_dataloaders
from engine import train
from utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    train_loader, test_loader, _ = build_dataloaders(
        ex=cfg["ex"],
        batch_size=cfg["batch_size"],
        test_size=cfg["test_size"],
        seed=cfg["split_seed"],
    )

    train(
        epochs=cfg["epoch"],
        train_loader=train_loader,
        test_loader=test_loader,
        save_path=cfg["save_dir"],
        ex=cfg["ex"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        huber_delta=cfg["huber_delta"],
    )


if __name__ == "__main__":
    main()
