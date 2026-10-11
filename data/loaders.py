"""Data loading utilities for RHYME."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np


class DeepfakeDataset(Dataset):
    """PyTorch Dataset for deepfake audio files."""

    def __init__(
        self,
        audio_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        sr: int = 16000,
        duration: float = 3.0,
    ):
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.max_length = int(sr * duration)

        if labels_file:
            self.audio_paths = self._load_from_labels(labels_file)
        else:
            self.audio_paths = list(self.audio_dir.glob("**/*.wav"))

    def _load_from_labels(self, labels_file: Union[str, Path]) -> List[Tuple]:
        """Load audio paths and labels from file."""
        import csv

        data = []
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'path': self.audio_dir / row['audio_path'],
                    'is_deepfake': int(row.get('is_deepfake', 0)),
                    'generator': int(row.get('generator', 0)),
                })
        return data

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get audio sample."""
        import librosa

        if isinstance(self.audio_paths[0], dict):
            item = self.audio_paths[idx]
            audio_path = item['path']
            labels = {
                'is_deepfake': item['is_deepfake'],
                'generator': item['generator'],
            }
        else:
            audio_path = self.audio_paths[idx]
            labels = {}

        # Load audio
        audio, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)

        # Pad or truncate
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]

        return {
            'audio': torch.from_numpy(audio).float(),
            'path': str(audio_path),
            **labels,
        }


class AudioDataLoader:
    """Wrapper for DeepfakeDataset with DataLoader."""

    def __init__(
        self,
        audio_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        sr: int = 16000,
        duration: float = 3.0,
    ):
        self.dataset = DeepfakeDataset(
            audio_dir=audio_dir,
            labels_file=labels_file,
            sr=sr,
            duration=duration,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)

    def __len__(self):
        """Get number of batches."""
        return len(self.loader)
