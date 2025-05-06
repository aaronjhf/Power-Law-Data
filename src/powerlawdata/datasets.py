import os
from pathlib import Path
import numpy as np

# TorchVision for CIFAR/MNIST
import torchvision.datasets as tvds
from torchvision import transforms

# HuggingFace for WikiText
from datasets import load_dataset

# 1. Default cache dir: env override or ~/.cache/powerlawdata
_DATA_ENV = "PLD_DATA"
_DEFAULT_CACHE = Path(os.getenv(_DATA_ENV, "~/.cache/powerlawdata")).expanduser()

def _get_data_dir(cfg: dict) -> Path:
    """
    Resolve config["data_dir"] or fall back to PLD_DATA / ~/.cache/powerlawdata.
    """
    if cfg and cfg.get("data_dir"):
        root = Path(cfg["data_dir"])
    else:
        root = _DEFAULT_CACHE
    root.mkdir(parents=True, exist_ok=True)
    return root

def load_cifar10(cfg: dict = None, flatten: bool = True) -> np.ndarray:
    """
    Returns (N, 3*32*32) float32 array in [0,1].
    Downloads into cache, never writes into the repo.
    """
    root = _get_data_dir(cfg)
    ds = tvds.CIFAR10(
        root=root,
        train=(cfg.get("split", "train") == "train"),
        download=True,
        transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    # stack into numpy
    X = np.stack([np.array(img).reshape(3*32*32) for img, _ in ds], axis=0)
    return X.astype(np.float32)

def load_mnist(cfg: dict = None, flatten: bool = True) -> np.ndarray:
    """
    Returns (N, 28*28) float32 array in [0,1].
    """
    root = _get_data_dir(cfg)
    ds = tvds.MNIST(
        root=root,
        train=(cfg.get("split", "train") == "train"),
        download=True,
        transform=transforms.ToTensor()
    )
    X = np.stack([np.array(img).reshape(28*28) for img, _ in ds], axis=0)
    return X.astype(np.float32)

def load_wikitext(cfg: dict = None, split: str = "train") -> list[str]:
    """
    Returns a list of raw text lines from WikiText-103.
    Cached by HuggingFace Datasets under HF_DATASETS_CACHE.
    """
    # Optionally override the HF cache via env var in your shell:
    #   export HF_DATASETS_CACHE=~/.cache/hf_datasets
    ds = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split=split,
        cache_dir=str(_get_data_dir(cfg) / "hf_datasets")
    )
    # return list of strings
    return ds["text"]

# Dispatch helper if you want a single factory
def get_dataset(name: str, cfg: dict):
    name = name.lower()
    if name == "cifar10":
        return load_cifar10(cfg, flatten=cfg.get("flatten", True))
    if name == "mnist":
        return load_mnist(cfg, flatten=cfg.get("flatten", True))
    if name.startswith("wikitext"):
        return load_wikitext(cfg, split=cfg.get("split", "train"))
    raise ValueError(f"Unsupported dataset: {name}")
