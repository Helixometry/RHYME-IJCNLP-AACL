"""Audio preprocessing utilities."""

import numpy as np
import librosa
from typing import Optional, Tuple


class AudioPreprocessor:
    """Audio preprocessing pipeline."""

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def load_audio(self, audio_path: str, mono: bool = True) -> np.ndarray:
        """Load audio file."""
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=mono)
        return audio

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc

    def extract_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

    def normalize(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features."""
        if mean is None:
            mean = features.mean(axis=1, keepdims=True)
        if std is None:
            std = features.std(axis=1, keepdims=True) + 1e-8

        normalized = (features - mean) / std
        return normalized, mean, std

    def process(self, audio_path: str) -> dict:
        """Complete preprocessing pipeline."""
        audio = self.load_audio(audio_path)

        features = {
            'mfcc': self.extract_mfcc(audio),
            'melspectrogram': self.extract_melspectrogram(audio),
        }

        return features
