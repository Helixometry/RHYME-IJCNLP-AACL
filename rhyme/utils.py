"""Utility functions for RHYME."""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Union, Optional, Tuple


def load_audio(
    audio_path: Union[str, Path],
    sr: int = 16000,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load audio file using librosa.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        mono: Convert to mono

    Returns:
        Audio tensor
    """
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=mono)
    return torch.from_numpy(audio).float()


def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    checkpoint_path: Union[str, Path],
    config: Optional[dict] = None,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class AverageMeter:
    """Keep track of average value."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
