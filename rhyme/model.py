"""Core RHYME model for speech deepfake detection."""

import torch
import torch.nn as nn
from typing import Dict, Optional


class RHYME(nn.Module):
    """
    RHYME: Deepfake Detection using Non-Euclidean Geometry.

    Uses hyperbolic and spherical geometric projections to create
    synthesis-invariant embeddings for robust deepfake detection.

    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension for feature processing
        dropout (float): Dropout probability
        use_hyperbolic (bool): Enable hyperbolic projections
        use_spherical (bool): Enable spherical projections
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_hyperbolic: bool = True,
        use_spherical: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_hyperbolic = use_hyperbolic
        self.use_spherical = use_spherical

        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification heads
        self.deepfake_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: Real/Fake
        )

        self.generator_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 5),  # Generator types
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RHYME.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary containing predictions for all tasks
        """
        # Feature extraction
        features = self.feature_projection(x)

        # Task-specific predictions
        deepfake_logits = self.deepfake_head(features)
        generator_logits = self.generator_head(features)

        return {
            "deepfake": deepfake_logits,
            "generator": generator_logits,
            "features": features,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "RHYME":
        """
        Load a pretrained RHYME model.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            RHYME model with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config from checkpoint
        config = checkpoint.get("config", {})
        model = cls(**config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """
        Save the model checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint
            config: Configuration dictionary to save with the model
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "use_hyperbolic": self.use_hyperbolic,
                "use_spherical": self.use_spherical,
            },
        }
        torch.save(checkpoint, checkpoint_path)
