"""Multi-encoder feature fusion for RHYME."""

import torch
import torch.nn as nn
from typing import List, Optional


class MultiEncoderFusion(nn.Module):
    """
    Fuse embeddings from multiple pretrained speech encoders.

    Supports WavLM, UniSpeech, HuBERT, XLS-R and other encoders.

    Args:
        encoder_dims (List[int]): Dimensions of encoder outputs
        output_dim (int): Output feature dimension
        fusion_type (str): 'concat', 'add', or 'attention'
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        encoder_dims: List[int],
        output_dim: int = 1024,
        fusion_type: str = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_encoders = len(encoder_dims)

        # Project each encoder's output
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for dim in encoder_dims
        ])

        # Fusion layer
        if fusion_type == "concat":
            total_dim = output_dim * len(encoder_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "add":
            self.fusion = nn.Identity()
        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, encoder_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse outputs from multiple encoders.

        Args:
            encoder_outputs: List of encoder outputs

        Returns:
            Fused feature embedding
        """
        # Project each encoder output
        projected = [
            proj(enc_out)
            for proj, enc_out in zip(self.projections, encoder_outputs)
        ]

        # Fuse projections
        if self.fusion_type == "concat":
            fused = torch.cat(projected, dim=-1)
            output = self.fusion(fused)
        elif self.fusion_type == "add":
            output = sum(projected) / len(projected)
        elif self.fusion_type == "attention":
            stacked = torch.stack(projected, dim=1)  # (batch, num_encoders, dim)
            attn_out, _ = self.attention(stacked, stacked, stacked)
            output = attn_out.mean(dim=1)

        return output


class EncoderWrapper(nn.Module):
    """
    Wrapper for pretrained speech encoders.

    Args:
        model_name (str): Name of pretrained model ('wavlm', 'unispeech', 'hubert', 'xlsr')
        freeze (bool): Freeze encoder weights
    """

    def __init__(self, model_name: str = "wavlm", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        if model_name == "wavlm":
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained("microsoft/wavlm-large")
                self.output_dim = 1024
            except ImportError:
                raise ImportError("Install transformers: pip install transformers")

        elif model_name == "unispeech":
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained("microsoft/unispeech-sat-large")
                self.output_dim = 1024
            except ImportError:
                raise ImportError("Install transformers: pip install transformers")

        elif model_name == "hubert":
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained("facebook/hubert-large-ls960")
                self.output_dim = 1024
            except ImportError:
                raise ImportError("Install transformers: pip install transformers")

        elif model_name == "xlsr":
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained("facebook/xlsr-300m")
                self.output_dim = 1024
            except ImportError:
                raise ImportError("Install transformers: pip install transformers")

        else:
            raise ValueError(f"Unknown encoder: {model_name}")

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from audio.

        Args:
            audio: Audio tensor of shape (batch_size, sample_length)

        Returns:
            Embeddings of shape (batch_size, seq_len, output_dim)
        """
        with torch.no_grad() if self.freeze else torch.enable_grad():
            output = self.encoder(audio, output_hidden_states=True)
            # Return last hidden state
            return output.last_hidden_state

    def get_output_dim(self) -> int:
        """Get output dimension of encoder."""
        return self.output_dim
