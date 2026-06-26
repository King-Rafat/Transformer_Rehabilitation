"""Tiny config helper: load YAML, then let argparse flags override fields."""
import argparse

import yaml


def load_config(default_path="configs/default.yaml"):
    parser = argparse.ArgumentParser(description="Point Cloud Transformer for Rehabilitation")
    parser.add_argument("--config", type=str, default=default_path, help="path to YAML config")
    parser.add_argument("--ex", type=str, help="exercise/dataset name, e.g. Kimore_ex5")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--epoch", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--save_dir", type=str, help="directory to save checkpoints")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path for eval")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config values with any CLI flags that were provided.
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value

    return cfg
