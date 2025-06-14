import os
import sys
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from train import train
from .config import ShakuhachiConfig


def train_shakuhachi(cfg: Optional[ShakuhachiConfig] = None) -> None:
    """Train a classifier for shakuhachi notation symbols."""
    cfg = cfg or ShakuhachiConfig()
    train('NN', 'hog', 'shakuhachi_model', dataset_path=cfg.dataset_path)


if __name__ == "__main__":
    train_shakuhachi()

